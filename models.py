"""
models.py -- single source of truth for every NeuralICU architecture.

All models are DP-SGD (Opacus) compatible, so the SAME network is used for
centralized, local-only, FedAvg and DP runs (the privacy cost is then measured
against a fair non-private baseline):
  * no BatchNorm (per-timestep LayerNorm only; GroupNorm would break TCN causality)
  * no weight_norm, no bare nn.Parameter outside supported layers
  * attention is built from nn.Linear + F.scaled_dot_product_attention
  * positional encoding is a registered *buffer* (not trainable)

INPUT CONTRACT  (B, T, F)  with F = 3 * n_var, channels laid out as
    [ values (n_var) | missing_flags (n_var) | recency (n_var) ]
  values  : forward-filled / imputed + standardized
  flags   : 1 = value was missing at this step (set flag_means_missing=False if 1 = observed)
  recency : hours since last measurement (clipped / scaled, >= 0)
TCN-only and TCN-Transformer treat all F channels as plain inputs; GRU-D uses
the 3-block layout to build its mask and decay.

OUTPUT  (B,)  raw logit (use BCEWithLogitsLoss; sigmoid for probability).

NOTE: these are reference implementations. If your published TCN / TCN-Transformer
differ (kernel size, widths, pooling), pass the same values via **cfg or edit the
classes here ONCE -- then every experiment inherits the change. Re-run
save_predictions.py afterwards and confirm the canonical test AUROC is reproduced.
"""
import math

import torch
# --- compat shim: some torch builds (esp. Windows/older) lack torch.distributed.fsdp.FSDPModule,
# which Opacus >=1.6 references. A never-matching dummy class is harmless for single-process training.
try:
    import torch.distributed.fsdp as _fsdp
    if not hasattr(_fsdp, "FSDPModule"):
        class FSDPModule:  # noqa: N801
            pass
        _fsdp.FSDPModule = FSDPModule
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each time step on a (B, C, T) tensor.
    Unlike GroupNorm it does not pool statistics across time, so the TCN stays causal;
    unlike BatchNorm it has no cross-sample statistics, so it is DP-SGD compatible."""

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def _group_norm(channels: int, groups: int = 4) -> nn.Module:  # kept name for compatibility
    return ChannelNorm(channels)


# --------------------------------------------------------------------------- TCN
class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)

    def forward(self, x):  # (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.n1 = _group_norm(c_out)
        self.c2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.n2 = _group_norm(c_out)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(F.relu(self.n1(self.c1(x))))
        h = self.drop(F.relu(self.n2(self.c2(h))))
        return F.relu(h + self.res(x))


class TCNEncoder(nn.Module):
    def __init__(self, n_features, hidden=64, levels=3, kernel_size=3, dropout=0.1):
        super().__init__()
        blocks, c = [], n_features
        for i in range(levels):
            blocks.append(TCNBlock(c, hidden, kernel_size, 2 ** i, dropout))
            c = hidden
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden

    def forward(self, x):  # (B,T,F) -> (B,T,H)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class TCNOnly(nn.Module):
    def __init__(self, n_features, hidden=64, levels=3, kernel_size=3, dropout=0.1, **_):
        super().__init__()
        self.enc = TCNEncoder(n_features, hidden, levels, kernel_size, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.enc(x)[:, -1]  # causal -> last step has seen the whole window
        return self.head(h).squeeze(-1)


# ------------------------------------------------------------ TCN + Transformer
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk, self.p = n_heads, d_model // n_heads, attn_dropout
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x):  # (B,T,D)
        B, T, D = x.shape
        split = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            split(self.q(x)), split(self.k(x)), split(self.v(x)),
            dropout_p=self.p if self.training else 0.0)
        return self.o(a.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult, dropout, attn_dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, attn_dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff_mult * d_model)
        self.ff2 = nn.Linear(ff_mult * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        return x + self.drop(self.ff2(self.drop(F.gelu(self.ff1(self.n2(x))))))


class TCNTransformer(nn.Module):
    """TCN encoder -> (optional PE) -> N attention blocks -> last-step head.
    pe: 'sinusoidal' | 'none'  (ablation switch for the positional-encoding question)
    """

    def __init__(self, n_features, seq_len=12, hidden=64, levels=3, kernel_size=3,
                 dropout=0.1, attn_dropout=0.1, n_heads=2, n_layers=1, ff_mult=2,
                 pe="sinusoidal", **_):
        super().__init__()
        self.enc = TCNEncoder(n_features, hidden, levels, kernel_size, dropout)
        self.use_pe = pe == "sinusoidal"
        if self.use_pe:
            pos = torch.arange(seq_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
            pe_t = torch.zeros(seq_len, hidden)
            pe_t[:, 0::2] = torch.sin(pos * div)
            pe_t[:, 1::2] = torch.cos(pos * div)[:, : hidden // 2]
            self.register_buffer("pe", pe_t.unsqueeze(0), persistent=False)
        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden, n_heads, ff_mult, dropout, attn_dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.enc(x)
        if self.use_pe:
            h = h + self.pe[:, : h.shape[1]]
        h = self.norm(self.blocks(h))
        return self.head(h[:, -1]).squeeze(-1)


# -------------------------------------------------------------------------- GRU-D
class GRUD(nn.Module):
    """GRU-D style cell (Che et al. 2018), simplified for DP compatibility:
    decays are full Linear layers (original uses diagonal weights) and the input
    is assumed already forward-filled, so x_last == x and x_mean == 0 (standardized).
    """

    def __init__(self, n_features, hidden=64, dropout=0.1, flag_means_missing=True, **_):
        super().__init__()
        assert n_features % 3 == 0, "GRU-D expects [values|flags|recency] = 3*n_var channels"
        self.v = n_features // 3
        self.hidden = hidden
        self.flag_means_missing = flag_means_missing
        self.decay_x = nn.Linear(self.v, self.v)
        self.decay_h = nn.Linear(self.v, hidden)
        self.gates = nn.Linear(2 * self.v + hidden, 2 * hidden)
        self.cand = nn.Linear(2 * self.v + hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        B, T, _ = x.shape
        val, flag, delta = x[..., : self.v], x[..., self.v: 2 * self.v], x[..., 2 * self.v:]
        mask = 1.0 - flag if self.flag_means_missing else flag
        h = x.new_zeros(B, self.hidden)
        for t in range(T):
            d = delta[:, t]
            gx = torch.exp(-F.relu(self.decay_x(d)))
            gh = torch.exp(-F.relu(self.decay_h(d)))
            m = mask[:, t]
            x_hat = m * val[:, t] + (1 - m) * gx * val[:, t]
            h = gh * h
            inp = torch.cat([x_hat, m, h], dim=-1)
            r, u = torch.chunk(torch.sigmoid(self.gates(inp)), 2, dim=-1)
            c = torch.tanh(self.cand(torch.cat([x_hat, m, r * h], dim=-1)))
            h = (1 - u) * h + u * c
        return self.head(self.drop(h)).squeeze(-1)


# ------------------------------------------------------------------------ factory
MODELS = {"tcn_only": TCNOnly, "tcn_transformer": TCNTransformer, "grud": GRUD}


def build_model(name, n_features, seq_len=12, **cfg):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](n_features=n_features, seq_len=seq_len, **cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dp_validation_errors(model):
    """Empty list == Opacus accepts this model unchanged."""
    from opacus.validators import ModuleValidator
    return ModuleValidator.validate(model, strict=False)
