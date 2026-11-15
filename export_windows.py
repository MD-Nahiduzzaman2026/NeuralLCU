"""
export_windows.py -- turn the project's frozen cache into windows.npz for fedavg.py / dp_sweep.py.

Uses common.load_cache() exactly like save_predictions.py, so the windows, order and split are
IDENTICAL to the XGBoost / TCN runs (train 748,287 / val 158,101 / test 164,486).

Run from the folder that contains common.py and config_physionet2019.py:

    python export_windows.py                  # inspect + export
    python export_windows.py --inspect        # only print what the cache contains

WHAT IT NEEDS (federated clients = hospital systems, PhysioNet 2019 set A vs set B):
  a per-window hospital label, or a per-window patient id from which it can be derived.
  Keys are searched as  site_{split} / hospital_{split} / set_{split}  then
  patient_id_{split} / pid_{split} / patient_ids_{split} / ids_{split}   (split = train|val|test).
  Override with --site-key "mysite_{}" or --pid-key "mypid_{}".

  If a patient id is found, hospital is derived as: numeric part >= 100000 -> 'B', else 'A'
  (PhysioNet 2019 files p000001.. = set A, p100001.. = set B). CHECK the printed counts: set A and
  set B should each hold roughly half of the patients. If ids were re-indexed 0..N this rule is wrong.

If the cache has neither, the script stops and lists the keys it found. Then the cache builder
(train_eval_physionet2019.py) must also save the per-window patient id / set label.
"""
import argparse
import re

import numpy as np

SITE_NAMES = ["site_{}", "hospital_{}", "set_{}"]
PID_NAMES = ["patient_id_{}", "pid_{}", "patient_ids_{}", "ids_{}", "groups_{}"]


def _keys(c):
    return list(c.keys()) if hasattr(c, "keys") else list(c.files)


def _site_from_pid(pids):
    out = []
    for p in pids:
        m = re.findall(r"\d+", str(p))
        v = int(m[-1]) if m else -1
        out.append("B" if v >= 100000 else "A")
    return np.array(out)


def find_site(c, split, site_fmt, pid_fmt):
    keys = _keys(c)
    for fmt in ([site_fmt] if site_fmt else SITE_NAMES):
        k = fmt.format(split)
        if k in keys:
            return np.asarray(c[k]).astype(str), f"{k} (as given)"
    for fmt in ([pid_fmt] if pid_fmt else PID_NAMES):
        k = fmt.format(split)
        if k in keys:
            return _site_from_pid(np.asarray(c[k])), f"{k} (derived: id>=100000 -> B)"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="windows.npz")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--site-key", default=None)
    ap.add_argument("--pid-key", default=None)
    ap.add_argument("--boundaries-json", default=None,
                    help="output of find_site_boundary.py: label rows before/after the detected boundary as set A / set B "
                         "(INFERRED from row order, not verified ids)")
    ap.add_argument("--synthetic-clients", type=int, default=0,
                    help="FALLBACK when the cache has no hospital/patient id: split each split into K contiguous "
                         "row blocks as simulated clients. These are NOT hospitals -- label them as simulated clients.")
    a = ap.parse_args()

    from common import load_cache
    c = load_cache()
    print("cache contents:")
    for k in _keys(c):
        v = c[k]
        print(f"   {k:28s} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', type(v).__name__)}")
    if a.inspect:
        X, y = np.asarray(c["X_train"]), np.asarray(c["y_train"])
        n = len(y)
        print("\nrow-order diagnostic (train, 10 consecutive blocks): does row order carry hospital structure?")
        print("   block  prevalence  mean_missing_rate")
        for i in range(10):
            s = slice(i * n // 10, (i + 1) * n // 10)
            print(f"   {i:5d}  {y[s].mean():10.4f}  {X[s][:, :, 2::3].mean():17.4f}")
        print("   A sharp step between blocks suggests rows are ordered by hospital; flat values mean no usable order.")
        return

    pack = {}
    for sp in ("train", "val", "test"):
        X = np.asarray(c[f"X_{sp}"], dtype=np.float32)
        y = np.asarray(c[f"y_{sp}"]).astype(np.float32)
        assert X.ndim == 3 and X.shape[1:] == (12, 36), f"{sp}: expected (N,12,36), got {X.shape}"
        assert len(X) == len(y)
        assert set(np.unique(y)) <= {0.0, 1.0}, f"{sp}: labels must be 0/1"
        assert np.isfinite(X).all(), f"{sp}: NaN/inf in X"
        site, how = find_site(c, sp, a.site_key, a.pid_key)
        if site is None and a.boundaries_json:
            import json
            b = json.load(open(a.boundaries_json))[sp]
            assert b, f"no boundary found for {sp}; rerun find_site_boundary.py"
            site = np.where(np.arange(len(y)) < b["idx"], "A", "B")
            how = f"INFERRED from row order: rows < {b['idx']} = A, rest = B (not verified ids)"
        if site is None and a.synthetic_clients > 0:
            K = a.synthetic_clients
            site = np.array([f"C{min(i * K // len(y), K - 1)}" for i in range(len(y))])
            how = f"SIMULATED: {K} contiguous row blocks (NOT hospitals)"
        if site is None:
            raise SystemExit(
                f"\nNo per-window hospital or patient id found for '{sp}'. Cache keys are listed above.\n"
                "Federated learning needs a hospital label per window (PhysioNet set A / set B).\n"
                "Either save patient ids in the cache builder (train_eval_physionet2019.py) or pass the right\n"
                "key with --site-key / --pid-key. Send me common.py + the cache-building script and I will add it.")
        assert len(site) == len(y), f"{sp}: site length {len(site)} != {len(y)}"
        print(f"{sp}: hospital label from {how}")
        site_is_sim = how.startswith("SIMULATED")
        site_is_inf = how.startswith("INFERRED")
        pack[f"X_{sp}"], pack[f"y_{sp}"], pack[f"site_{sp}"] = X, y, site

    pack["partition"] = np.array("inferred_row_order_AB" if a.boundaries_json and site_is_inf else
                                 "simulated_contiguous" if a.synthetic_clients > 0 and site_is_sim else "hospital")
    np.savez(a.out, **pack)           # uncompressed: much faster to load, ~2 GB
    print(f"\nwrote {a.out}")
    for sp in ("train", "val", "test"):
        y, site = pack[f"y_{sp}"], pack[f"site_{sp}"]
        print(f"\n{sp}: n={len(y)}  prevalence={y.mean():.4f}")
        for s in np.unique(site):
            m = site == s
            print(f"   site {s}: n={m.sum()}  positives={int(y[m].sum())}  prevalence={y[m].mean():.4f}")
    print("\nCHECK: n(train/val/test) = 748287 / 158101 / 164486, and the two sites are of similar size.")


if __name__ == "__main__":
    main()
