"""
fedavg.py -- centralized / local-only / FedAvg / FedProx / DP-FedAvg under ONE protocol.

Every run: same frozen split (you supply it), same model factory (models.py),
same optimizer, same early-stopping rule (validation AUROC), test set touched once.

Provenance (answers the "88 tensors differ" audit finding):
  * fedavg() is a single, unit-tested function (test_fed.py)
  * every round logs client sizes, client-state SHA-256, global-state SHA-256
  * best global checkpoint is saved and its hash is written to the results row

DP (Opacus RDP accountant):
  * one PrivacyEngine PER CLIENT, created once and kept for all rounds
    (re-creating it each round would under-report epsilon)
  * Poisson sampling, per-sample clipping, sigma chosen by
    make_private_with_epsilon for the FULL budget of rounds*local_epochs epochs
  * epsilon after each round is read from the accountant, not computed by hand
  * record-level DP (one window = one record). A patient has many windows, so this
    is NOT patient-level DP -- state that in the paper.

DATA (npz, produced by your existing pipeline; already standardized):
  X_train, y_train, site_train, X_val, y_val, site_val, X_test, y_test, site_test
  X is (N, 12, 36) float32 -- the cached windows, channel c = columns [3c, 3c+1, 3c+2] = [value, trend, flag].
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from fed_models import build_model, count_params


# ----------------------------------------------------------------------- utilities
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_state(state):
    return {k: v.detach().clone().cpu() for k, v in state.items()}


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fedavg(states, sizes):
    """Sample-count-weighted average of state dicts.
    Floating tensors: weighted mean (float64 accumulation, cast back).
    Integer buffers (e.g. num_batches_tracked): taken from client 0 (documented policy).
    """
    assert len(states) == len(sizes) and len(states) > 0
    total = float(sum(sizes))
    out = {}
    for k, t0 in states[0].items():
        if t0.is_floating_point():
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for s, n in zip(states, sizes):
                acc += s[k].to(torch.float64) * (n / total)
            out[k] = acc.to(t0.dtype)
        else:
            out[k] = t0.clone()
    return out


def log_jsonl(path, rec):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


# ----------------------------------------------------------------------- metrics
def ece_score(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def safe_auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def predict(model, X, device="cpu", bs=8192):
    model.eval().to(device)
    out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def full_metrics(model, data, device="cpu"):
    """Raw + isotonic-calibrated metrics (calibrator fit on VALIDATION only),
    plus per-site AUROC with class counts (single-class site -> NaN, never 0.5)."""
    pv = predict(model, data["val"]["X"], device)
    pt = predict(model, data["test"]["X"], device)
    yt = data["test"]["y"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, data["val"]["y"])
    pc = iso.predict(pt)
    res = dict(
        test_auroc=safe_auroc(yt, pt),
        test_auprc=float(average_precision_score(yt, pt)),
        brier=float(brier_score_loss(yt, pt)),
        ece=ece_score(yt, pt),
        brier_cal=float(brier_score_loss(yt, pc)),
        ece_cal=ece_score(yt, pc),
        brier_ref=float(yt.mean() * (1 - yt.mean())),  # no-skill Brier -> skill score
        n_test=int(len(yt)), n_pos=int(yt.sum()),
        pred_std=float(pt.std()),
    )
    per_site = {}
    for s in np.unique(data["test"]["site"]):
        m = data["test"]["site"] == s
        per_site[str(s)] = dict(n=int(m.sum()), n_pos=int(yt[m].sum()),
                                auroc=safe_auroc(yt[m], pt[m]),
                                pred_std=float(pt[m].std()))
    res["per_site"] = per_site
    return res, pt, pc


# ------------------------------------------------------------------------- data
def load_npz(path, seq_len=12, layout="blocks"):
    """layout='blocks'   : flat X = [values T*V | flags T*V | recency T*V]  (e.g. 144|144|144)
       layout='stepmajor': flat X = T steps, each [values V | flags V | recency V]
       3-D X is used as-is.  VERIFY this against your feature builder before trusting results."""
    if not path or not os.path.exists(path):
        raise SystemExit(f"Data file not found: {path!r}. Create it first with export_windows.py, "
                         f"then run:  python fedavg.py --data windows.npz --model tcn_only")
    z = np.load(path, allow_pickle=True)

    def prep(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            return X
        n, f = X.shape
        v = f // (3 * seq_len)
        assert v * 3 * seq_len == f, f"cannot reshape {f} features with seq_len={seq_len}"
        if layout == "blocks":
            parts = [X[:, i * seq_len * v:(i + 1) * seq_len * v].reshape(n, seq_len, v) for i in range(3)]
            return np.concatenate(parts, axis=2)
        return X.reshape(n, seq_len, 3 * v)

    d = {"partition": str(z["partition"]) if "partition" in z.files else "unknown"}
    for sp in ("train", "val", "test"):
        d[sp] = dict(X=prep(z[f"X_{sp}"]), y=np.asarray(z[f"y_{sp}"], dtype=np.float32),
                     site=np.asarray(z[f"site_{sp}"]).astype(str))
    return d


def site_view(data, split, site):
    m = data[split]["site"] == site
    return data[split]["X"][m], data[split]["y"][m]


# ----------------------------------------------------------------------- training
def make_opt(model, a):
    return torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)


def train_epoch(model, opt, loader, loss_fn, device, anchor=None, mu=0.0, gclip=0.0):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb) if len(xb) else None
        if loss is None:       # empty Poisson batch (DP): still step so accounting matches
            loss = sum(p.sum() * 0 for p in model.parameters())
        loss.backward()
        if gclip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gclip)
        if anchor is not None and mu > 0:           # FedProx: grad += mu * (w - w_global)
            for (k, p) in model.named_parameters():
                if p.grad is not None:
                    p.grad.add_(mu * (p.data - anchor[k].to(p.device)))
        opt.step()
        tot += float(loss) * len(xb)
        n += len(xb)
    return tot / max(n, 1)


def loader_for(X, y, bs, seed, shuffle=True):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g)


def val_auroc(model, data, device):
    return safe_auroc(data["val"]["y"], predict(model, data["val"]["X"], device))


def result_row(base, metrics):
    row = dict(base)
    ps = metrics.pop("per_site")
    row.update(metrics)
    row["per_site"] = json.dumps(ps)
    return row


def append_csv(path, row):
    """Append one result row. Modes produce different columns (e.g. DP adds sigma/delta), so the file is
    rewritten with the UNION of all columns every time -- rows always line up with the header."""
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(dict(row))
    cols = []
    for r in rows:
        for k in r:
            if k is not None and k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------- run modes
def run_centralized(data, a, seed):
    set_seed(seed)
    dev = a.device
    model = build_model(a.model, a.n_features, a.seq_len, arch=a.arch).to(dev)
    opt = make_opt(model, a)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(a.pw, device=dev))
    best, best_state, best_ep, bad = -1, None, 0, 0
    for ep in range(1, a.max_epochs + 1):
        ld = loader_for(data["train"]["X"], data["train"]["y"], a.bs, seed * 1000 + ep)
        loss = train_epoch(model, opt, ld, loss_fn, dev, gclip=a.gclip)
        v = val_auroc(model, data, dev)
        log_jsonl(a.log, dict(mode="centralized", seed=seed, epoch=ep, loss=loss, val_auroc=v))
        if v > best:
            best, best_state, best_ep, bad = v, clone_state(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, dict(best_round=best_ep, val_auroc=best, state_hash=state_hash(best_state), eps_spent_max=float("inf"))


def run_local_only(data, a, seed):
    """One model per site, each early-stopped on ITS OWN validation data; evaluated per-site on the
    matching test slice and pooled by concatenating predictions. Returns pooled metrics dict directly."""
    set_seed(seed)
    dev = a.device
    sites = sorted(np.unique(data["train"]["site"]))
    pt = np.zeros(len(data["test"]["y"]), dtype=np.float64)
    pv = np.zeros(len(data["val"]["y"]), dtype=np.float64)
    hashes, best_eps = {}, {}
    for s in sites:
        Xtr, ytr = site_view(data, "train", s)
        Xva, yva = site_view(data, "val", s)
        model = build_model(a.model, a.n_features, a.seq_len, arch=a.arch).to(dev)
        opt = make_opt(model, a)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(a.pw, device=dev))
        best, bst, bep, bad = -1, None, 0, 0
        for ep in range(1, a.max_epochs + 1):
            train_epoch(model, opt, loader_for(Xtr, ytr, a.bs, seed * 1000 + ep), loss_fn, dev, gclip=a.gclip)
            v = safe_auroc(yva, predict(model, Xva, dev))
            v = -0.5 if math.isnan(v) else v
            if v > best:
                best, bst, bep, bad = v, clone_state(model.state_dict()), ep, 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(bst)
        hashes[s], best_eps[s] = state_hash(bst), bep
        pt[data["test"]["site"] == s] = predict(model, data["test"]["X"][data["test"]["site"] == s], dev)
        pv[data["val"]["site"] == s] = predict(model, Xva, dev)
    return pt, pv, dict(best_round=json.dumps(best_eps), state_hash=json.dumps(hashes), eps_spent_max=float("inf"))


def run_federated(data, a, seed, mu=0.0, eps=float("inf")):
    set_seed(seed)
    dev = a.device
    sites = sorted(np.unique(data["train"]["site"]))
    views = {s: site_view(data, "train", s) for s in sites}
    sizes = [len(views[s][1]) for s in sites]
    use_dp = math.isfinite(eps)
    delta = a.delta if a.delta else 1.0 / (10 * min(sizes))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(a.pw, device=dev))

    global_model = build_model(a.model, a.n_features, a.seq_len, arch=a.arch).to(dev)
    gstate = clone_state(global_model.state_dict())

    dp = {}
    if use_dp:
        from opacus import PrivacyEngine
        for i, s in enumerate(sites):
            m = build_model(a.model, a.n_features, a.seq_len, arch=a.arch).to(dev)
            o = make_opt(m, a)
            dl = loader_for(*views[s], a.bs, seed * 100 + i)
            pe = PrivacyEngine(accountant="rdp")
            m, o, dl = pe.make_private_with_epsilon(
                module=m, optimizer=o, data_loader=dl,
                epochs=a.rounds * a.local_epochs, target_epsilon=eps, target_delta=delta,
                max_grad_norm=a.clip, poisson_sampling=True)
            dp[s] = (m, o, dl, pe)

    best, best_state, best_round, bad = -1, None, 0, 0
    eps_hist = []
    for r in range(1, a.rounds + 1):
        cstates, closs = [], []
        for i, s in enumerate(sites):
            if use_dp:
                m, o, dl, pe = dp[s]
                m._module.load_state_dict(gstate)          # new global weights, SAME engine/accountant
                for _ in range(a.local_epochs):
                    loss = train_epoch(m, o, dl, loss_fn, dev)
                cstates.append(clone_state(m._module.state_dict()))
            else:
                m = build_model(a.model, a.n_features, a.seq_len, arch=a.arch).to(dev)
                m.load_state_dict(gstate)
                o = make_opt(m, a)
                for e in range(a.local_epochs):
                    ld = loader_for(*views[s], a.bs, seed * 100000 + r * 100 + i * 10 + e)
                    loss = train_epoch(m, o, ld, loss_fn, dev, anchor=gstate if mu > 0 else None, mu=mu, gclip=a.gclip)
                cstates.append(clone_state(m.state_dict()))
            closs.append(loss)
        gstate = fedavg(cstates, sizes)
        global_model.load_state_dict(gstate)
        v = val_auroc(global_model, data, dev)
        eps_now = max(dp[s][3].accountant.get_epsilon(delta) for s in sites) if use_dp else float("inf")
        eps_hist.append(eps_now)
        log_jsonl(a.log, dict(mode="fedprox" if mu > 0 else "fedavg", seed=seed, round=r, sites=sites, sizes=sizes,
                              client_loss=closs, client_hashes=[state_hash(c)[:12] for c in cstates],
                              global_hash=state_hash(gstate), val_auroc=v, eps_max=eps_now, delta=delta,
                              sigma=(dp[sites[0]][1].noise_multiplier if use_dp else None)))
        if v > best:
            best, best_state, best_round, bad = v, clone_state(gstate), r, 0
        else:
            bad += 1
            if bad >= a.patience_rounds:
                break
    global_model.load_state_dict(best_state)
    info = dict(best_round=best_round, val_auroc=best, state_hash=state_hash(best_state),
                eps_spent_max=eps_hist[-1], eps_at_best=eps_hist[best_round - 1], delta=delta if use_dp else None,
                sigma=dp[sites[0]][1].noise_multiplier if use_dp else None)
    return global_model, info


# ------------------------------------------------------------------------ driver
def run_one(data, a, mode, seed, eps=float("inf")):
    t0 = time.time()
    private = mode == "fedavg" and math.isfinite(eps)
    if getattr(a, "paper_config", False) and not private:
        ytr = data["train"]["y"]                       # paper recipe: neg/pos ratio from training labels
        a.pw = float(max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0))
    else:                                              # DP (or default): fixed constant, no privacy cost
        a.pw = a.pos_weight
    base = dict(model=a.model, mode=mode, seed=seed, eps_target=eps, rounds=a.rounds, local_epochs=a.local_epochs,
                clip=a.clip, bs=a.bs, lr=a.lr, wd=a.wd, arch=a.arch, pos_weight=a.pw,
                partition=data.get("partition", "unknown"))
    if mode == "local":
        pt, pv, info = run_local_only(data, a, seed)
        yt, yvv = data["test"]["y"], data["val"]["y"]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yvv)
        pc = iso.predict(pt)
        m = dict(test_auroc=safe_auroc(yt, pt), test_auprc=float(average_precision_score(yt, pt)),
                 brier=float(brier_score_loss(yt, pt)), ece=ece_score(yt, pt),
                 brier_cal=float(brier_score_loss(yt, pc)), ece_cal=ece_score(yt, pc),
                 brier_ref=float(yt.mean() * (1 - yt.mean())), n_test=len(yt), n_pos=int(yt.sum()), pred_std=float(pt.std()),
                 per_site={str(s): dict(n=int((data["test"]["site"] == s).sum()),
                                        auroc=safe_auroc(yt[data["test"]["site"] == s], pt[data["test"]["site"] == s]))
                           for s in np.unique(data["test"]["site"])})
        np.save(os.path.join(a.out, f"pred_{a.model}_{mode}_s{seed}.npy"), pt)
    else:
        if mode == "centralized":
            model, info = run_centralized(data, a, seed)
        else:
            model, info = run_federated(data, a, seed, mu=a.mu if mode == "fedprox" else 0.0, eps=eps)
        m, pt, _ = full_metrics(model, data, a.device)
        torch.save(model.state_dict(), os.path.join(a.out, f"ckpt_{a.model}_{mode}_eps{eps}_s{seed}.pt"))
        np.save(os.path.join(a.out, f"pred_{a.model}_{mode}_eps{eps}_s{seed}.npy"), pt)
        base["n_params"] = count_params(model)
    row = result_row({**base, **info}, m)
    row["seconds"] = round(time.time() - t0, 1)
    append_csv(os.path.join(a.out, "results_canonical.csv"), row)
    print(f"[{mode} eps={eps} seed={seed}] test AUROC {row['test_auroc']:.4f}  AUPRC {row['test_auprc']:.4f}"
          f"  eps_spent={info.get('eps_spent_max')}  hash={str(info.get('state_hash'))[:12]}")
    return row


def add_args(p):
    p.add_argument("--data", required=False)
    p.add_argument("--layout", default="blocks", choices=["blocks", "stepmajor"])
    p.add_argument("--model", default="tcn_only", choices=["tcn_only", "tcn_transformer", "grud"])
    p.add_argument("--n_features", type=int, default=0, help="channels per step (auto from data if 0)")
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    p.add_argument("--modes", nargs="+", default=["centralized", "local", "fedavg"],
                   choices=["centralized", "local", "fedavg", "fedprox"])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--max_epochs", type=int, default=30, help="centralized / local-only epoch cap")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience_rounds", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--arch", default="reference", choices=["reference", "paper"],
                   help="reference = fed_models.py networks; paper = project's original models.py made DP-compatible")
    p.add_argument("--paper_config", action="store_true",
                   help="use TRAIN_CONFIG from config_physionet2019.py (AdamW, lr, wd, bs, patience, grad clip) "
                        "and the paper's neg/pos pos_weight for NON-private runs")
    p.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay")
    p.add_argument("--gclip", type=float, default=0.0, help="grad-norm clip for non-DP training (0=off)")
    p.add_argument("--pos_weight", type=float, default=5.0,
                   help="fixed constant (NOT computed from client data, so it costs no privacy)")
    p.add_argument("--mu", type=float, default=0.01, help="FedProx strength")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.0, help="0 -> 1/(10*smallest client)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs")
    return p


def apply_paper_config(a):
    """Copy the project's own training recipe so the centralized run is comparable to the canonical one."""
    if not getattr(a, "paper_config", False):
        return a
    import sys
    from config_physionet2019 import TRAIN_CONFIG as T
    given = {x.lstrip("-").split("=")[0] for x in sys.argv if x.startswith("--")}   # explicit CLI flags win
    for flag, key in (("lr", "learning_rate"), ("wd", "weight_decay"), ("bs", "batch_size"),
                      ("patience", "early_stopping_patience"), ("max_epochs", "max_epochs"), ("gclip", "gradient_clip_norm")):
        if flag not in given:
            setattr(a, flag, T[key])
    print(f"[paper_config] lr={a.lr} wd={a.wd} bs={a.bs} patience={a.patience} max_epochs={a.max_epochs} gclip={a.gclip}")
    return a


def main():
    a = apply_paper_config(add_args(argparse.ArgumentParser()).parse_args())
    os.makedirs(a.out, exist_ok=True)
    a.log = os.path.join(a.out, "rounds.jsonl")
    data = load_npz(a.data, a.seq_len, a.layout)
    a.n_features = a.n_features or data["train"]["X"].shape[2]
    for s in sorted(np.unique(data["train"]["site"])):
        y = data["train"]["y"][data["train"]["site"] == s]
        print(f"site {s}: n_train={len(y)} prevalence={y.mean():.4f}")
    for seed in a.seeds:
        for mode in a.modes:
            run_one(data, a, mode, seed)


if __name__ == "__main__":
    main()
