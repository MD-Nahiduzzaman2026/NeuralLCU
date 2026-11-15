"""
find_site_boundary.py -- test whether the cached rows are ordered by hospital (PhysioNet set A, then set B)
and, if so, locate the A/B boundary in each split.

Idea: if rows were built by walking the files in order (all of set A, then all of set B) and the
train/val/test split kept that order, then label prevalence and some feature distributions shift ONCE,
at the same relative position in every split. A change-point (CUSUM) search on 25 per-row signals
(label, 12 per-channel missingness, 12 per-channel value means) finds that point; if many independent
signals agree on the same position, and train/val/test agree with each other, row order carries hospital.

    python find_site_boundary.py            # reads windows.npz, writes boundaries.json

This is INFERENCE, not ground truth. Use it to run the federated experiments with 'inferred hospital'
clients and say so in the paper, or (better) save real patient ids in the cache builder and re-export.
The permutation ratio below only ranks signals; consecutive windows of one patient are correlated, so
it is not a p-value.
"""
import argparse
import json

import numpy as np


def cusum(s, rng, n_perm=5):
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    sd = s.std() or 1.0
    k = np.arange(1, n)
    norm = np.sqrt(k * (n - k) / n)

    def stat(v):
        c = np.cumsum(v - v.mean())[:-1]
        t = np.abs(c) / (sd * norm)
        return t

    t = stat(s)
    i = int(t.argmax())
    null = max(stat(rng.permutation(s)).max() for _ in range(n_perm))
    return i + 1, float(t[i]), float(t[i] / max(null, 1e-9))


def row_signals(X, y):
    sig = {"label": y.astype(np.float64)}
    for c in range(X.shape[2] // 3):
        sig[f"missing_ch{c}"] = X[:, :, 3 * c + 2].mean(axis=1)
        sig[f"value_ch{c}"] = X[:, :, 3 * c].mean(axis=1)
    return sig


def analyse(X, y, name, rng, min_ratio=3.0):
    n = len(y)
    rows = []
    for k, s in row_signals(X, y).items():
        idx, T, ratio = cusum(s, rng)
        rows.append((k, idx, idx / n, T, ratio))
    rows.sort(key=lambda r: -r[4])
    print(f"\n[{name}] n={n}   strongest change points (signal, row, fraction, stat, stat/permutation-null)")
    for r in rows[:8]:
        print(f"   {r[0]:12s} {r[1]:8d}  {r[2]:.3f}  {r[3]:8.1f}  {r[4]:6.1f}x")
    good = [r for r in rows if r[4] >= min_ratio]
    if len(good) < 3:
        print(f"   -> fewer than 3 signals show a clear step: NO evidence of hospital ordering in {name}")
        return None
    fr = np.array([r[2] for r in good])
    med = float(np.median(fr))
    near = int((np.abs(fr - med) < 0.02).sum())
    print(f"   -> {len(good)} signals show a step; {near} of them within 2% of median position {med:.3f}")
    return dict(idx=int(round(med * n)), frac=med, agree=near, n_signals=len(good), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="windows.npz")
    ap.add_argument("--out", default="boundaries.json")
    a = ap.parse_args()
    z = np.load(a.data, allow_pickle=True)
    rng = np.random.default_rng(0)
    res = {}
    for sp in ("train", "val", "test"):
        res[sp] = analyse(np.asarray(z[f"X_{sp}"]), np.asarray(z[f"y_{sp}"]), sp, rng)
    fr = [r["frac"] for r in res.values() if r]
    print("\n=== verdict ===")
    if len(fr) == 3 and max(fr) - min(fr) < 0.03 and all(r["agree"] >= 3 for r in res.values()):
        print(f"Rows look ordered by hospital: boundary at {min(fr):.3f}-{max(fr):.3f} of rows in all three splits.")
        print("Inferred (not verified) set A = rows before the boundary, set B = rows after.")
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}   ->  python export_windows.py --boundaries-json {a.out}")
    else:
        print("No consistent boundary across train/val/test: do NOT treat row blocks as hospitals.")
        print("Save real patient ids in the cache builder instead (send common.py + the builder script).")


if __name__ == "__main__":
    main()
