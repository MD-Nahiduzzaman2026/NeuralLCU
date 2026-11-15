"""
Fix 3 (corrected): XGBoost baseline on the SAME 100-patient MIMIC-III Demo
cohort, split by true SUBJECT_ID (real patient) rather than STAY_ID, with
diagnostics that surface a bad/imbalanced split immediately, and a
patient-level bootstrap CI (window-level bootstrapping is invalid here
because windows from the same patient are highly correlated).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

import sys

RANDOM_STATE = int(sys.argv[1]) if len(sys.argv) > 1 else 42
WINDOW = 12

df = pd.read_csv("hourly_features_labeled.csv")

if "SUBJECT_ID" not in df.columns:
    raise ValueError(
        "SUBJECT_ID column missing -- rerun the updated build_labeled_features.py first."
    )

feature_cols = [c for c in df.columns if c not in ("STAY_ID", "HOUR", "SEPSIS", "SUBJECT_ID")]
print(f"Using {len(feature_cols)} feature columns: {feature_cols}")

# ---------------------------------------------------------------------
# Patient-wise split by SUBJECT_ID (fixes stay-level leakage)
# ---------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

# Diagnostics: septic PATIENT counts, not just row counts, in each split
train_patient_labels = df_train.groupby("SUBJECT_ID")["SEPSIS"].max()
test_patient_labels = df_test.groupby("SUBJECT_ID")["SEPSIS"].max()

print(f"Train: {df_train['SUBJECT_ID'].nunique()} patients, "
      f"{train_patient_labels.sum()} septic ({100*train_patient_labels.mean():.1f}%)")
print(f"Test:  {df_test['SUBJECT_ID'].nunique()} patients, "
      f"{test_patient_labels.sum()} septic ({100*test_patient_labels.mean():.1f}%)")

if test_patient_labels.sum() < 5:
    print("\n*** WARNING: fewer than 5 septic patients in the test set. ***")
    print("*** Any AUROC computed here will be extremely high-variance and ***")
    print("*** should NOT be reported as a point estimate without this caveat. ***")


def create_windows(df_subset, feature_cols, window=WINDOW):
    X, y, subject_ids = [], [], []
    for sid in df_subset["STAY_ID"].unique():
        patient = (
            df_subset[df_subset["STAY_ID"] == sid]
            .sort_values("HOUR")
            .reset_index(drop=True)
        )
        if len(patient) < window + 1:
            continue
        feats = patient[feature_cols].values
        labels = patient["SEPSIS"].values
        subj = patient["SUBJECT_ID"].iloc[0]
        for i in range(len(patient) - window):
            X.append(feats[i : i + window].flatten())
            y.append(labels[i + window])
            subject_ids.append(subj)
    return np.array(X), np.array(y), np.array(subject_ids)


X_train, y_train, _ = create_windows(df_train, feature_cols)
X_test, y_test, test_subject_ids = create_windows(df_test, feature_cols)

print(f"\nTrain windows: {X_train.shape}, positive rate: {y_train.mean():.3f}")
print(f"Test windows:  {X_test.shape}, positive rate: {y_test.mean():.3f}")

if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
    raise ValueError(
        "One class is entirely missing from train or test after windowing -- "
        "try a different random_state, or this cohort is too small to split reliably."
    )

model = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_pred)
auprc = average_precision_score(y_test, y_pred)

print("\n=== POINT ESTIMATE ===")
print(f"XGBoost AUROC: {auroc:.4f}")
print(f"XGBoost AUPRC: {auprc:.4f}")
print(f"Test patients: {len(np.unique(test_subject_ids))} "
      f"({int(test_patient_labels.sum())} septic)")


def bootstrap_auroc_ci_patient_level(y_true, y_score, subject_ids, n_boot=1000, random_state=RANDOM_STATE):
    """Resamples whole PATIENTS (not individual windows), since windows from
    the same patient are correlated and window-level bootstrapping understates
    true uncertainty."""
    rng = np.random.RandomState(random_state)
    unique_subjects = np.unique(subject_ids)
    scores = []
    for _ in range(n_boot):
        sampled_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        mask = np.concatenate([np.where(subject_ids == s)[0] for s in sampled_subjects])
        if len(np.unique(y_true[mask])) < 2:
            continue
        scores.append(roc_auc_score(y_true[mask], y_score[mask]))
    if len(scores) < 100:
        print(f"WARNING: only {len(scores)}/{n_boot} bootstrap resamples had both classes present -- "
              f"CI below is unreliable; the test set has too few septic patients.")
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return lower, upper, len(scores)


ci_low, ci_high, n_valid = bootstrap_auroc_ci_patient_level(y_test, y_pred, test_subject_ids)
print(f"Patient-level bootstrap 95% CI ({n_valid} valid resamples): {ci_low:.4f}-{ci_high:.4f}")
