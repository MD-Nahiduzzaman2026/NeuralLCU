"""
STEP 2 - calibration, operating points and alert burden from the predictions
saved by save_predictions.py (no model is retrained).

Run from the same folder as save_predictions.py:

    python operating_points.py --model xgboost --n-seeds 5

Reads  PRED_DIR/{model}_seed{seed}.npz   (val_probs, test_probs)
Labels from load_cache() (y_val, y_test).

Method (all thresholds are chosen on VALIDATION data only, then applied to test):
  * Calibration: Platt scaling (logistic regression on the logit of the score)
    fitted on validation, applied to test. Monotone, so it does not change
    AUROC or the ranking, and it creates no ties. Brier, ECE and the reliability
    curve are reported for the raw and the calibrated scores.
  * Operating points: for each target validation sensitivity (default 0.30, 0.50,
    0.70 = Conservative / Balanced / Screening) the threshold is the highest
    score that still reaches that sensitivity on validation.
  * On test: sensitivity, specificity, PPV, NPV, alert rate per 1,000 ICU-hours
    (every window is one patient-hour, hourly stride), and lift over prevalence.

Outputs in OUT/operating_points/:
  per_seed_{model}.csv, summary_{model}.csv, summary_{model}.md,
  reliability_{model}.png
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common import OUT, PRED_DIR, load_cache
from config_physionet2019 import SEED_CONFIG

MODES = {"Conservative": 0.30, "Balanced": 0.50, "Screening": 0.70}


def logit(p, eps=1e-7):
    p = np.clip(p.astype(np.float64), eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt_fit_apply(val_p, y_val, test_p):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit(val_p), y_val)
    return lr.predict_proba(logit(test_p))[:, 1]


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def threshold_for_sensitivity(y, s, target):
    """Highest threshold whose sensitivity on (y, s) is >= target."""
    pos = np.sort(s[y == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return pos[max(k, 0)]


def metrics_at(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                alerts_per_1000_hr=1000.0 * pred.mean(), tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    c = load_cache()
    y_val, y_test = c["y_val"].astype(int), c["y_test"].astype(int)
    prev = y_test.mean()
    outdir = f"{OUT}/operating_points"
    os.makedirs(outdir, exist_ok=True)

    rows, rel = [], []
    for seed in SEED_CONFIG["seeds"][: args.n_seeds]:
        path = f"{PRED_DIR}/{args.model}_seed{seed}.npz"
        if not os.path.exists(path):
            print(f"missing {path}; skipped")
            continue
        z = np.load(path)
        vp, tp_ = z["val_probs"], z["test_probs"]
        cal = platt_fit_apply(vp, y_val, tp_)
        base = dict(
            seed=seed,
            auroc=roc_auc_score(y_test, tp_),
            auprc=average_precision_score(y_test, tp_),
            prevalence=prev,
            brier_raw=brier_score_loss(y_test, np.clip(tp_, 0, 1)),
            brier_platt=brier_score_loss(y_test, cal),
            ece_raw=ece(y_test, np.clip(tp_, 0, 1)),
            ece_platt=ece(y_test, cal),
        )
        for mode, target in MODES.items():
            thr = threshold_for_sensitivity(y_val, vp, target)
            m = metrics_at(y_test, tp_, thr)
            m["lift_over_prevalence"] = m["ppv"] / prev
            m["val_threshold"] = float(thr)
            rows.append({**base, "mode": mode, "target_val_sens": target, **m})
        bins = np.minimum((cal * 10).astype(int), 9)
        for b in range(10):
            mk = bins == b
            if mk.any():
                rel.append(dict(seed=seed, bin=b, mean_pred=cal[mk].mean(),
                                obs=y_test[mk].mean(), n=int(mk.sum())))

    if not rows:
        raise SystemExit("no prediction files found - run save_predictions.py first")

    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/per_seed_{args.model}.csv", index=False)

    keep = ["sensitivity", "specificity", "ppv", "npv", "alerts_per_1000_hr",
            "lift_over_prevalence"]
    g = df.groupby("mode", sort=False)
    summ = g[keep].agg(["mean", "std"])
    summ.columns = ["_".join(k) for k in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(f"{outdir}/summary_{args.model}.csv", index=False)

    glob = df.drop_duplicates("seed")[["auroc", "auprc", "brier_raw", "brier_platt",
                                         "ece_raw", "ece_platt"]].agg(["mean", "std"])

    with open(f"{outdir}/summary_{args.model}.md", "w") as f:
        f.write(f"Model: {args.model}; seeds: {df.seed.nunique()}; test prevalence {prev:.4f}\n\n")
        f.write("| Mode | Sens | Spec | PPV | Alerts / 1,000 ICU-hr | Lift |\n|---|---|---|---|---|---|\n")
        for _, r in summ.iterrows():
            f.write(f"| {r['mode']} | {r.sensitivity_mean:.3f} ± {r.sensitivity_std:.3f} | "
                    f"{r.specificity_mean:.3f} ± {r.specificity_std:.3f} | "
                    f"{r.ppv_mean:.3f} ± {r.ppv_std:.3f} | "
                    f"{r.alerts_per_1000_hr_mean:.0f} ± {r.alerts_per_1000_hr_std:.0f} | "
                    f"{r.lift_over_prevalence_mean:.1f}x |\n")
        f.write("\nDiscrimination / calibration (mean ± SD over seeds):\n\n")
        f.write(glob.round(4).to_markdown() + "\n")
        f.write("\nCaveats: frozen patient split, so SD reflects training noise only. "
                "Alerts are per window (patient-hour); consecutive alerts for one patient are "
                "not merged into episodes. Lead time and alerts per patient-day need patient "
                "ids and onset times, which the cached tensors do not contain.\n")
    print(open(f"{outdir}/summary_{args.model}.md").read())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = pd.DataFrame(rel).groupby("bin").agg(mean_pred=("mean_pred", "mean"),
                                                 obs=("obs", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, max(r.mean_pred.max(), r.obs.max()) * 1.05],
                [0, max(r.mean_pred.max(), r.obs.max()) * 1.05], "k--", lw=1, label="perfect")
        ax.plot(r.mean_pred, r.obs, "o-", label="Platt-calibrated")
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed sepsis rate")
        ax.set_title(f"{args.model}: reliability (test)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/reliability_{args.model}.png", dpi=200)
    except Exception as e:  # plotting is optional
        print("reliability plot skipped:", e)

    print(f"\nWritten to {outdir}")


if __name__ == "__main__":
    main()
