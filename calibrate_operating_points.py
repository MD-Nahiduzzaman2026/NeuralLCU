"""
STEP 2 - calibration, operating points and alert burden (Reviewer 2 #6).

Input : outputs/predictions/{model}_seed{seed}.npz  (from save_predictions.py)
Output: outputs/calibration_operating_points/
          calibration_summary.csv      before/after calibration, mean +- SD over seeds
          operating_points.csv         per model, seed, operating point
          operating_points_summary.csv mean +- SD over seeds (paste into Table 9)
          reliability_<model>.png      reliability diagram before/after
Run:
    python calibrate_operating_points.py --models xgboost tcn_only tcn_transformer \
        --method platt --max-fa-per-1000h 10 --screening-sensitivity 0.90

Protocol (everything chosen on VALIDATION, applied once to TEST):
  1. Calibrate scores on validation (Platt = logistic on logit(p); or isotonic).
  2. Choose thresholds on validation:
       Conservative: lowest threshold with <= X false alerts per 1,000 ICU-hours
                     (X is --max-fa-per-1000h; justify X from literature/clinicians).
       Balanced    : threshold maximising Youden J (sensitivity + specificity - 1).
       Screening   : highest threshold reaching --screening-sensitivity window sensitivity.
  3. Apply the three thresholds once to test and report metrics.

Alert rules (state these in Methods):
  * One prediction per ICU hour, made at the last observed hour (t).
  * An alert fires when the score >= threshold; after an alert the patient is
    silenced for --refractory hours (default 6).
  * Hours at or after the labelled onset are excluded from burden counts.
  * Alert is TRUE if onset follows within the 6 h horizon (lead = onset - t, 1..6 h);
    otherwise FALSE (this includes alerts on never-septic patients and alerts
    earlier than 6 h before onset). Lead time is therefore capped at 6 h by design.
  * False alerts per 1,000 ICU-hours = false alerts / eligible patient-hours * 1000.
  * PhysioNet 2019 onset labels are shifted relative to clinical onset, so lead
    time is relative to the labelled onset (state this caveat).
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common import HORIZON, OUT, PRED_DIR, ece_score, load_indexed_splits, patient_slices

OUTDIR = f"{OUT}/calibration_operating_points"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrator(method, val_p, y_val):
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val_p, y_val)
        return iso.predict
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(val_p)[:, None], y_val)
    return lambda p: lr.predict_proba(logit(p)[:, None])[:, 1]


def slope_intercept(y, p):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(p)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def prob_metrics(y, p):
    s, i = slope_intercept(y, p)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece_score(y, p),
            "cal_slope": s, "cal_intercept": i}


def window_counts(y, flag):
    tp = int(((flag == 1) & (y == 1)).sum()); fn = int(((flag == 0) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    return tp, fp, tn, fn


def window_metrics(y, s, thr):
    tp, fp, tn, fn = window_counts(y, (s >= thr).astype(int))
    d = lambda a, b: a / b if b else float("nan")
    return {"sensitivity": d(tp, tp + fn), "specificity": d(tn, tn + fp),
            "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn)}


def simulate_alerts(s, thr, idx, slices, refractory):
    """Alert-level evaluation, see module docstring."""
    pid, t, onset = idx["pid"], idx["t"], idx["onset"]
    n_alerts = n_false = n_true = 0
    hours = 0
    septic_patients = 0
    detected = 0
    leads = []
    for a, b in zip(*slices):
        o = onset[a]
        tt = t[a:b]
        eligible = np.ones(b - a, dtype=bool) if o < 0 else tt < o
        hours += int(eligible.sum())
        if o >= 0:
            septic_patients += 1
        flagged = np.flatnonzero((s[a:b] >= thr) & eligible)
        last = -10**9
        first_true = None
        for i in flagged:
            if tt[i] - last < refractory:
                continue
            last = tt[i]
            n_alerts += 1
            if o >= 0 and 1 <= o - tt[i] <= HORIZON:
                n_true += 1
                if first_true is None:
                    first_true = o - tt[i]
            else:
                n_false += 1
        if first_true is not None:
            detected += 1
            leads.append(first_true)
    return {
        "alerts": n_alerts, "false_alerts": n_false, "true_alerts": n_true,
        "icu_hours": hours,
        "false_alerts_per_1000h": 1000 * n_false / hours if hours else float("nan"),
        "alerts_per_patient_day": 24 * n_alerts / hours if hours else float("nan"),
        "patient_sensitivity": detected / septic_patients if septic_patients else float("nan"),
        "median_lead_h": float(np.median(leads)) if leads else float("nan"),
        "lead_iqr_low": float(np.percentile(leads, 25)) if leads else float("nan"),
        "lead_iqr_high": float(np.percentile(leads, 75)) if leads else float("nan"),
        "n_septic_patients": septic_patients, "n_detected": detected,
    }


def choose_thresholds(s_val, y_val, idx_val, slices_val, refractory, max_fa, sens_target):
    out = {}
    # Balanced: Youden J on windows
    grid = np.unique(np.quantile(s_val, np.concatenate([np.linspace(0, 0.95, 120),
                                                        1 - np.logspace(-1.3, -4, 80)])))
    best_j, best_t = -1, grid[0]
    for g in grid:
        m = window_metrics(y_val, s_val, g)
        j = m["sensitivity"] + m["specificity"] - 1
        if np.isfinite(j) and j > best_j:
            best_j, best_t = j, g
    out["Balanced"] = float(best_t)
    # Screening: highest threshold with sensitivity >= target
    ok = [g for g in grid if window_metrics(y_val, s_val, g)["sensitivity"] >= sens_target]
    out["Screening"] = float(max(ok)) if ok else float(grid[0])
    # Conservative: lowest threshold whose false alerts/1000h <= max_fa (bisection)
    lo, hi = 0, len(grid) - 1
    fa = lambda g: simulate_alerts(s_val, g, idx_val, slices_val, refractory)["false_alerts_per_1000h"]
    if fa(grid[hi]) > max_fa:
        out["Conservative"] = float(grid[hi])
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if fa(grid[mid]) <= max_fa:
                hi = mid
            else:
                lo = mid + 1
        out["Conservative"] = float(grid[lo])
    return out


def patient_bootstrap_ci(y, p, pid, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    sl = patient_slices(pid)
    starts, ends = sl
    n = len(starts)
    au, ap = [], []
    for _ in range(n_boot):
        pick = rng.randint(0, n, n)
        ii = np.concatenate([np.arange(starts[k], ends[k]) for k in pick])
        if y[ii].min() == y[ii].max():
            continue
        au.append(roc_auc_score(y[ii], p[ii])); ap.append(average_precision_score(y[ii], p[ii]))
    return (np.percentile(au, [2.5, 97.5]), np.percentile(ap, [2.5, 97.5]))


def reliability_plot(y, before, after, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for p, lab in [(before, "before"), (after, "after calibration")]:
        q = np.quantile(p, np.linspace(0, 1, 11))
        ids = np.clip(np.searchsorted(q[1:-1], p), 0, 9)
        xs = [p[ids == b].mean() for b in range(10) if (ids == b).any()]
        ys = [y[ids == b].mean() for b in range(10) if (ids == b).any()]
        ax.plot(xs, ys, marker="o", label=lab)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed event rate")
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xgboost", "tcn_only", "tcn_transformer"])
    ap.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    ap.add_argument("--max-fa-per-1000h", type=float, default=10.0)
    ap.add_argument("--screening-sensitivity", type=float, default=0.90)
    ap.add_argument("--refractory", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    cache, vi, ti = load_indexed_splits()
    y_val, y_test = cache["y_val"], cache["y_test"]
    sl_val, sl_test = patient_slices(vi["pid"]), patient_slices(ti["pid"])

    cal_rows, op_rows = [], []
    for model in args.models:
        files = sorted(glob.glob(f"{PRED_DIR}/{model}_seed*.npz"))
        if not files:
            print(f"no predictions for {model}, skipping"); continue
        for fi, f in enumerate(files):
            seed = int(f.split("seed")[-1].split(".")[0])
            d = np.load(f)
            vp, tp = d["val_probs"].astype(np.float64), d["test_probs"].astype(np.float64)
            calib = fit_calibrator(args.method, vp, y_val)
            vc, tc = calib(vp), calib(tp)
            for stage, p in [("raw", tp), ("calibrated", tc)]:
                cal_rows.append({"model": model, "seed": seed, "stage": stage, **prob_metrics(y_test, p)})
            if fi == 0:
                reliability_plot(y_test, tp, tc, f"{model} (seed {seed}, test)",
                                 f"{OUTDIR}/reliability_{model}.png")
                if args.bootstrap:
                    (alo, ahi), (plo, phi) = patient_bootstrap_ci(y_test, tc, ti["pid"], args.bootstrap)
                    print(f"{model} seed {seed}: patient-bootstrap 95% CI AUROC {alo:.4f}-{ahi:.4f}, "
                          f"AUPRC {plo:.4f}-{phi:.4f} (prevalence {y_test.mean():.4f})")
            thr = choose_thresholds(vc, y_val, vi, sl_val, args.refractory,
                                    args.max_fa_per_1000h, args.screening_sensitivity)
            for name in ["Conservative", "Balanced", "Screening"]:
                th = thr[name]
                row = {"model": model, "seed": seed, "operating_point": name, "threshold_calibrated_prob": th,
                       **{f"window_{k}": v for k, v in window_metrics(y_test, tc, th).items()},
                       **simulate_alerts(tc, th, ti, sl_test, args.refractory)}
                op_rows.append(row)
            print(f"{model} seed {seed}: thresholds " + ", ".join(f"{k}={v:.4f}" for k, v in thr.items()))

    cal = pd.DataFrame(cal_rows); ops = pd.DataFrame(op_rows)
    cal.to_csv(f"{OUTDIR}/calibration_by_seed.csv", index=False)
    ops.to_csv(f"{OUTDIR}/operating_points.csv", index=False)
    summ = cal.groupby(["model", "stage"]).agg(["mean", "std"]).drop(columns="seed")
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.to_csv(f"{OUTDIR}/calibration_summary.csv")
    num = ops.drop(columns=["seed"]).groupby(["model", "operating_point"]).agg(["mean", "std"])
    num.columns = ["_".join(c) for c in num.columns]
    num.to_csv(f"{OUTDIR}/operating_points_summary.csv")
    print(f"\nWrote results to {OUTDIR}")
    print(summ[[c for c in summ.columns if c.split("_")[0] in ("auroc", "auprc", "brier", "ece")]].round(4).to_string())


if __name__ == "__main__":
    main()
