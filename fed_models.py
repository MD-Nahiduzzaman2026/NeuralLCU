"""
fed_models.py -- reference architectures for the federated / DP experiments.
(Named fed_models.py on purpose: the project already has its own models.py with
build_model_factory; never overwrite that file.)

All models are DP-SGD (Opacus) compatible, so the SAME network is used for
centralized, local-only, FedAvg and DP runs (the privacy cost is then measured
against a fair non-private baseline):
  * no BatchNorm (per-timestep LayerNorm only; GroupNorm would break TCN causality)
  * no weight_norm, no bare nn.Parameter outside supported layers
  * attention is built from nn.Linear + F.scaled_dot_product_attention
  * positional encoding is a registered *buffer* (not trainable)

INPUT CONTRACT  (B, T=12, F=36)  -- exactly the cached windows used everywhere else in the project
    channel c (of 12) occupies columns 3c, 3c+1, 3c+2 = [value, trend, missing_flag]
    (value standardized with TRAIN stats; flag 1.0 = missing inside the window; see
    missingness_ablation.py: values = X[:, :, 0::3], missing = X[:, :, 2::3] > 0.5)
TCN-only and TCN-Transformer treat all 36 channels as plain inputs; GRU-D reads the value and flag
columns and derives time-since-last-observation itself.

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
    """GRU-D style cell (Che et al. 2018), simplified for DP compatibility: decays are Linear layers
    (original: diagonal) and the input is already standardized, so the empirical mean is 0.
    Uses the cached layout [value, trend, flag] per channel; recency is derived from the flags."""

    def __init__(self, n_features, seq_len=12, hidden=64, dropout=0.1, **_):
        super().__init__()
        assert n_features % 3 == 0, "expects 3 columns per channel"
        self.v, self.T, self.hidden = n_features // 3, seq_len, hidden
        self.decay_x = nn.Linear(self.v, self.v)
        self.decay_h = nn.Linear(self.v, hidden)
        self.gates = nn.Linear(2 * self.v + hidden, 2 * hidden)
        self.cand = nn.Linear(2 * self.v + hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        B, T, _ = x.shape
        val, flag = x[..., 0::3], x[..., 2::3]
        mask = 1.0 - flag                                   # 1 = observed
        h = x.new_zeros(B, self.hidden)
        delta = x.new_zeros(B, self.v)
        for t in range(T):
            m = mask[:, t]
            delta = (1.0 - m) * (delta + 1.0)               # hours since last observation
            d = delta / self.T
            gx = torch.exp(-F.relu(self.decay_x(d)))
            gh = torch.exp(-F.relu(self.decay_h(d)))
            x_hat = m * val[:, t] + (1 - m) * gx * val[:, t]
            h = gh * h
            inp = torch.cat([x_hat, m, h], dim=-1)
            r, u = torch.chunk(torch.sigmoid(self.gates(inp)), 2, dim=-1)
            c = torch.tanh(self.cand(torch.cat([x_hat, m, r * h], dim=-1)))
            h = (1 - u) * h + u * c
        return self.head(self.drop(h)).squeeze(-1)


# ------------------------------------------------------------------------ factory
MODELS = {"tcn_only": TCNOnly, "tcn_transformer": TCNTransformer, "grud": GRUD}


class _SqueezeOut(nn.Module):
    """Makes any (B,) / (B,1) logit output uniform (B,)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return o.reshape(o.shape[0], -1)[:, 0] if o.dim() > 1 else o


def build_paper_model(name, n_raw_channels=12, seed=0):
    """Build the architecture from the project's ORIGINAL models.py (build_model_factory), then make
    it Opacus-compatible (BatchNorm -> GroupNorm) so centralized / local / FedAvg / DP all use one
    network. NOTE: this swap changes the architecture slightly versus the paper's non-DP numbers --
    say so, and compare federated results only against the centralized run of THIS same network."""
    from models import build_model_factory  # the user's original file, not this one
    fac = build_model_factory(n_raw_channels=n_raw_channels)[name]
    try:
        m = fac(seed)
    except TypeError:
        m = fac()
    m = _SqueezeOut(m)
    from opacus.validators import ModuleValidator
    errs = ModuleValidator.validate(m, strict=False)
    if errs:
        print(f"[fed_models] making '{name}' DP-compatible; replaced: {[str(e)[:70] for e in errs]}")
        m = ModuleValidator.fix(m)
        left = ModuleValidator.validate(m, strict=False)
        if left:
            raise RuntimeError(f"model still not Opacus-compatible after fix: {left}")
    return m


def build_model(name, n_features, seq_len=12, arch="reference", seed=0, **cfg):
    if arch == "paper":
        return build_paper_model(name, cfg.pop("n_raw_channels", n_features // 3), seed)
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](n_features=n_features, seq_len=seq_len, **cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dp_validation_errors(model):
    """Empty list == Opacus accepts this model unchanged."""
    from opacus.validators import ModuleValidator
    return ModuleValidator.validate(model, strict=False)
