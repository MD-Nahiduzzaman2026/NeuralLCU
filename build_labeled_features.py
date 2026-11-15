"""
Same as before, but now also carries SUBJECT_ID (the real patient identifier)
through to the output file, so the downstream split can be done patient-wise
rather than stay-wise -- fixing a leakage risk (some patients have 2+ ICU stays).
"""

import pandas as pd

FEATURES_CSV = "Model train with demo/hourly_features.csv"
MIMIC_DIR = "mimic_demo/mimic-iii-clinical-database-demo-1.4"
OUTPUT_CSV = "hourly_features_labeled.csv"

SEPSIS_ICD9_CODES = {"99591", "99592", "78552"} | {
    f"038{d}" for d in ["0", "1", "2", "3", "40", "41", "42", "43", "44", "49", "8", "9"]
}

icustays = pd.read_csv(f"{MIMIC_DIR}/ICUSTAYS.csv")
diag = pd.read_csv(f"{MIMIC_DIR}/DIAGNOSES_ICD.csv", dtype={"icd9_code": str})

diag["is_sepsis_code"] = diag["icd9_code"].isin(SEPSIS_ICD9_CODES)
sepsis_hadm_ids = set(diag.loc[diag["is_sepsis_code"], "hadm_id"])

icustays["SEPSIS"] = icustays["hadm_id"].isin(sepsis_hadm_ids).astype(int)
label_map = icustays.set_index("icustay_id")["SEPSIS"]
subject_map = icustays.set_index("icustay_id")["subject_id"]  # <-- NEW: real patient ID

n_pos = label_map.sum()
n_total = len(label_map)
print(f"ICU stays: {n_total}, sepsis-positive: {n_pos} ({100*n_pos/n_total:.1f}%)")

df = pd.read_csv(FEATURES_CSV)
df = df.rename(columns={"ICUSTAY_ID": "STAY_ID", "HOUR_BIN": "HOUR"})

if "STAY_ID" not in df.columns:
    raise ValueError(f"Couldn't find ICUSTAY_ID/STAY_ID column. Columns: {list(df.columns)}")

df["SEPSIS"] = df["STAY_ID"].map(label_map)
df["SUBJECT_ID"] = df["STAY_ID"].map(subject_map)  # <-- NEW

n_missing = df["SEPSIS"].isna().sum()
if n_missing > 0:
    print(f"WARNING: {n_missing} rows had no matching ICUSTAY_ID (dropping them).")
    df = df.dropna(subset=["SEPSIS", "SUBJECT_ID"])

df["SEPSIS"] = df["SEPSIS"].astype(int)
df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(int)

# Diagnostic: how many real patients have multiple ICU stays in this cohort?
stays_per_patient = df.groupby("SUBJECT_ID")["STAY_ID"].nunique()
multi_stay_patients = (stays_per_patient > 1).sum()
print(f"Real patients: {df['SUBJECT_ID'].nunique()}, "
      f"of which {multi_stay_patients} have multiple ICU stays "
      f"(these MUST stay together in the same split)")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_CSV}: {df['SUBJECT_ID'].nunique()} patients, "
      f"{df['STAY_ID'].nunique()} stays, {len(df)} rows")