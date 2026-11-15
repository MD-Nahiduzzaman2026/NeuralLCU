"""
STEP 3 - controlled missingness ablation (Reviewer 2 #5).

Every strategy is trained and tested on EXACTLY the same patients, the same
windows (the frozen cache), with the SAME model and hyperparameters, so the
only thing that changes is how missing values are represented.

Strategies (12 h window, 12 channels):
  S1 mean_imputation        train-set mean fills every missing value
  S2 ffill_mean             forward fill inside the window, mean for leading gaps
  S3 ffill_mean+flags       S2 plus observed/missing flags
  S4 ffill_mean+flags+recency  S3 plus hours since last measurement
  S5 native_nan             no imputation; XGBoost routes missing values itself
  S6 current_pipeline       the 36-feature cache used in the paper (value, trend, flag)
  S7 missingness_only       flags + recency, NO measured values
Key comparisons: S3 vs S2 isolates the flag effect; S7 shows how much signal is
measurement frequency alone.

Each strategy is evaluated on (a) the full test set and (b) the complete-case
test subset (windows whose last hour has all 12 channels measured). Listwise
deletion is thereby reported as a separate analysis on the same models; it is
not used as a baseline for the flag effect.

Run (needs models.py for the XGBoost settings; falls back to defaults if absent):
    python missingness_ablation.py --n-seeds 5
Outputs: outputs/missingness_ablation/{by_seed.csv, summary.csv, paired_differences.csv}

Note: the split is frozen, so seeds vary only the model's own randomness. The
paired confidence intervals therefore describe training noise, not
sampling uncertainty over patients; say so in the paper.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common import N_CH, NORM_STATS_PATH, OUT, WINDOW, ece_score, load_cache
from config_physionet2019 import RAW_CHANNELS, SEED_CONFIG

OUTDIR = f"{OUT}/missingness_ablation"


def get_model(name, seed):
    try:
        from models import build_model_factory
        m = build_model_factory(n_raw_channels=len(RAW_CHANNELS))[name]
        try:
            return m(seed)
        except TypeError:
            return m()
    except Exception as e:  # models.py not importable: transparent fallback
        print(f"WARNING: could not use models.py ({e}); falling back to default settings. "
              f"Results will not be comparable to the paper's table.")
        if name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", random_state=seed,
                                 eval_metric="logloss")
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)


def denormalize(X):
    st = json.load(open(NORM_STATS_PATH))
    mean, std = np.array(st["mean"], dtype=np.float32), np.array(st["std"], dtype=np.float32)
    return X * std + mean


def split_parts(X):
    Xd = denormalize(X)
    vals = Xd[:, :, 0::3].copy()
    mask = Xd[:, :, 2::3] > 0.5          # True = missing inside the window
    return vals, mask


def ffill_mean(vals, mask, mean):
    obs = ~mask
    idx = np.where(obs, np.arange(WINDOW)[None, :, None], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    out = np.take_along_axis(vals, np.clip(idx, 0, None), axis=1)
    out = np.where(idx < 0, mean[None, None, :], out).astype(np.float32)
    recency = np.where(idx < 0, WINDOW, (WINDOW - 1) - idx).astype(np.float32)
    return out, recency


def build_features(strategy, vals, mask, mean, X_cached):
    flat = lambda a: a.reshape(len(a), -1).astype(np.float32)
    if strategy == "S6_current_pipeline":
        return flat(X_cached)
    if strategy == "S1_mean_imputation":
        return flat(np.where(mask, mean[None, None, :], vals))
    ff, rec = ffill_mean(vals, mask, mean)
    if strategy == "S2_ffill_mean":
        return flat(ff)
    if strategy == "S3_ffill_mean+flags":
        return np.concatenate([flat(ff), flat(mask.astype(np.float32))], axis=1)
    if strategy == "S4_ffill_mean+flags+recency":
        return np.concatenate([flat(ff), flat(mask.astype(np.float32)), flat(rec)], axis=1)
    if strategy == "S5_native_nan":
        return flat(np.where(mask, np.nan, vals))
    if strategy == "S7_missingness_only":
        return np.concatenate([flat(mask.astype(np.float32)), flat(rec)], axis=1)
    raise ValueError(strategy)


STRATEGIES = ["S1_mean_imputation", "S2_ffill_mean", "S3_ffill_mean+flags",
              "S4_ffill_mean+flags+recency", "S5_native_nan", "S6_current_pipeline",
              "S7_missingness_only"]


def platt(val_p, y_val, test_p):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg(val_p), y_val)
    return lr.predict_proba(lg(test_p))[:, 1]


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest"])
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    c = load_cache()
    y_tr, y_va, y_te = c["y_train"], c["y_val"], c["y_test"]
    vals_tr, mask_tr = split_parts(c["X_train"])
    obs = ~mask_tr
    mean = ((vals_tr * obs).sum((0, 1)) / np.maximum(obs.sum((0, 1)), 1)).astype(np.float32)
    vals_va, mask_va = split_parts(c["X_val"])
    vals_te, mask_te = split_parts(c["X_test"])
    cc = ~mask_te[:, -1, :].any(axis=1)     # complete-case: all channels measured in the last hour
    print(f"Windows: train {len(y_tr)}, val {len(y_va)}, test {len(y_te)}; "
          f"complete-case test windows {int(cc.sum())} ({100*cc.mean():.2f}%), positives {int(y_te[cc].sum())}")
    print(f"Fraction of values missing inside windows (train): {100*mask_tr.mean():.1f}%")

    seeds = SEED_CONFIG["seeds"][: args.n_seeds]
    rows = []
    for strat in args.strategies:
        if strat == "S5_native_nan" and args.model != "xgboost":
            continue
        Xtr = build_features(strat, vals_tr, mask_tr, mean, c["X_train"])
        Xva = build_features(strat, vals_va, mask_va, mean, c["X_val"])
        Xte = build_features(strat, vals_te, mask_te, mean, c["X_test"])
        print(f"\n{strat}: {Xtr.shape[1]} features")
        for seed in seeds:
            np.random.seed(seed)
            m = get_model(args.model, seed)
            if args.model == "xgboost":
                pos = max(y_tr.sum(), 1); neg = max(len(y_tr) - y_tr.sum(), 1)
                m.set_params(scale_pos_weight=neg / pos)
            m.fit(Xtr, y_tr)
            pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            pc = platt(pv, y_va, pt)
            r = {"strategy": strat, "seed": seed, "n_features": Xtr.shape[1],
                 "n_train": len(y_tr), "n_test": len(y_te),
                 "auroc": roc_auc_score(y_te, pt), "auprc": average_precision_score(y_te, pt),
                 "brier_calibrated": brier_score_loss(y_te, pc), "ece_calibrated": ece_score(y_te, pc),
                 "n_cc_test": int(cc.sum()), "auroc_cc": safe_auc(y_te[cc], pt[cc]),
                 "auprc_cc": average_precision_score(y_te[cc], pt[cc]) if cc.any() and y_te[cc].max() else float("nan")}
            rows.append(r)
            print(f"  seed {seed}: AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  "
                  f"Brier(cal) {r['brier_calibrated']:.4f}  complete-case AUROC {r['auroc_cc']:.4f}")
        del Xtr, Xva, Xte
        pd.DataFrame(rows).to_csv(f"{OUTDIR}/by_seed.csv", index=False)

    df = pd.DataFrame(rows)
    num = ["auroc", "auprc", "brier_calibrated", "ece_calibrated", "auroc_cc", "auprc_cc"]
    summ = df.groupby("strategy")[num].agg(["mean", "std"])
    summ.columns = ["_".join(x) for x in summ.columns]
    summ.insert(0, "n_test_windows", len(y_te)); summ.insert(1, "n_cc_test_windows", int(cc.sum()))
    summ.to_csv(f"{OUTDIR}/summary.csv")
    print("\n=== SUMMARY (mean over seeds) ===")
    print(summ.round(4).to_string())

    pairs = [("S2_ffill_mean", "S1_mean_imputation", "forward fill vs mean imputation"),
             ("S3_ffill_mean+flags", "S2_ffill_mean", "FLAG EFFECT (flags vs same imputation without)"),
             ("S4_ffill_mean+flags+recency", "S3_ffill_mean+flags", "recency features on top of flags"),
             ("S5_native_nan", "S2_ffill_mean", "native NaN vs forward fill"),
             ("S6_current_pipeline", "S3_ffill_mean+flags", "paper pipeline vs S3"),
             ("S7_missingness_only", "S2_ffill_mean", "missingness-only vs values-only")]
    prow = []
    for a, b, label in pairs:
        A = df[df.strategy == a].sort_values("seed"); B = df[df.strategy == b].sort_values("seed")
        if len(A) == 0 or len(B) == 0 or len(A) != len(B):
            continue
        for metric in ["auroc", "auprc", "auroc_cc"]:
            d = A[metric].values - B[metric].values
            d = d[~np.isnan(d)]
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            tcrit = stats.t.ppf(0.975, len(d) - 1)
            try:
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = float("nan")
            prow.append({"comparison": label, "a": a, "b": b, "metric": metric, "mean_diff": d.mean(),
                         "ci_low": d.mean() - tcrit * se, "ci_high": d.mean() + tcrit * se,
                         "wilcoxon_p": p, "n_pairs": len(d)})
    pdf = pd.DataFrame(prow)
    pdf.to_csv(f"{OUTDIR}/paired_differences.csv", index=False)
    print("\n=== PAIRED DIFFERENCES (a minus b) ===")
    print(pdf.round(4).to_string(index=False))
    print("\nNOTE: frozen split => differences reflect model-training noise only; "
          "report the small seed count and this caveat.")


if __name__ == "__main__":
    main()
