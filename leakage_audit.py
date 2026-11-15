"""
Leakage audit -- Steps 1 and 5.
Checks patient-ID overlap across splits and duplicate rows across splits.

Usage:
    python leakage_audit.py --split patient_split_physionet2019.json --data Dataset.csv

Expects:
  --split : JSON with keys "train", "val", "test" -> lists of patient IDs
            (adjust KEY_TRAIN/KEY_VAL/KEY_TEST below if your JSON uses
            different key names -- check with: python -c "import json;
            print(json.load(open('patient_split_physionet2019.json')).keys())")
  --data  : CSV with a patient-id column and a timestamp/hour column
            (adjust ID_COL / TIME_COL below to match your actual columns)
"""

import argparse
import json
import sys
import pandas as pd

KEY_TRAIN = "train"
KEY_VAL = "val"
KEY_TEST = "test"

ID_COL = "Patient_ID"     # change if your CSV uses a different column name
TIME_COL = "Hour"         # change if your CSV uses a different column name


def load_split(path):
    with open(path) as f:
        split = json.load(f)

    missing = [k for k in (KEY_TRAIN, KEY_VAL, KEY_TEST) if k not in split]
    if missing:
        print(f"[FATAL] split JSON missing keys: {missing}")
        print(f"        keys found: {list(split.keys())}")
        print("        -> edit KEY_TRAIN/KEY_VAL/KEY_TEST at top of this script to match.")
        sys.exit(1)

    train_ids = set(split[KEY_TRAIN])
    val_ids = set(split[KEY_VAL])
    test_ids = set(split[KEY_TEST])
    return train_ids, val_ids, test_ids


def step1_patient_overlap(train_ids, val_ids, test_ids):
    print("=" * 70)
    print("STEP 1 -- Patient-ID overlap across splits")
    print("=" * 70)

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    print(f"train n={len(train_ids)}  val n={len(val_ids)}  test n={len(test_ids)}")
    print(f"train ∩ val   : {len(tv)} patients {'-- LEAK' if tv else '-- clean'}")
    print(f"train ∩ test  : {len(tt)} patients {'-- LEAK' if tt else '-- clean'}")
    print(f"val ∩ test    : {len(vt)} patients {'-- LEAK' if vt else '-- clean'}")

    if tv or tt or vt:
        print()
        print("Overlapping patient IDs (first 20 of each):")
        if tv:
            print(f"  train∩val : {sorted(list(tv))[:20]}")
        if tt:
            print(f"  train∩test: {sorted(list(tt))[:20]}")
        if vt:
            print(f"  val∩test  : {sorted(list(vt))[:20]}")
        print()
        print("[RESULT] LEAK CONFIRMED at patient-ID level.")
        print("         Same patient's windows exist in >1 split.")
        print("         Fix: rebuild split so each patient_id appears in exactly one split,")
        print("         then rebuild ALL windows from that corrected split (not just re-filter")
        print("         existing windows -- if windows were built before the split, redo that")
        print("         step too, see Step 1b below).")
        return False
    else:
        print()
        print("[RESULT] No patient-ID overlap at the split-file level. Clean so far.")
        print("         Still need Step 1b: confirm windows were built AFTER splitting patients,")
        print("         not before (row-level split of pre-built windows can still leak).")
        return True


def step1b_window_build_order(data_path, train_ids, val_ids, test_ids):
    print()
    print("=" * 70)
    print("STEP 1b -- Row-level check: do dataset rows for each patient")
    print("           actually land in exactly one split?")
    print("=" * 70)
    try:
        df = pd.read_csv(data_path, usecols=[ID_COL])
    except Exception as e:
        print(f"[SKIP] could not read data file with column '{ID_COL}': {e}")
        print(f"       edit ID_COL at top of script to match your actual column name.")
        return

    all_data_ids = set(df[ID_COL].unique())
    split_ids = train_ids | val_ids | test_ids
    unassigned = all_data_ids - split_ids
    unknown_in_split = split_ids - all_data_ids

    print(f"Unique patient IDs in raw data file: {len(all_data_ids)}")
    print(f"Unique patient IDs across split file: {len(split_ids)}")
    print(f"Patients in data but NOT in any split: {len(unassigned)} "
          f"{'-- these rows get silently dropped or (worse) silently included somewhere' if unassigned else '-- clean'}")
    print(f"Patients in split file but NOT in data: {len(unknown_in_split)} "
          f"{'-- stale split file, rebuild it' if unknown_in_split else '-- clean'}")


def step5_duplicate_rows(data_path, train_ids, val_ids, test_ids):
    print()
    print("=" * 70)
    print("STEP 5 -- Duplicate rows across splits (row-level leak, not just ID)")
    print("=" * 70)
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"[SKIP] could not read data file: {e}")
        return

    if ID_COL not in df.columns:
        print(f"[SKIP] column '{ID_COL}' not found in data. Columns present: {list(df.columns)}")
        print(f"       edit ID_COL at top of script.")
        return

    key_cols = [ID_COL] + ([TIME_COL] if TIME_COL in df.columns else [])
    if TIME_COL not in df.columns:
        print(f"[WARN] TIME_COL '{TIME_COL}' not found -- deduping on {ID_COL} only "
              f"(less precise, patient-level not row-level).")

    train_df = df[df[ID_COL].isin(train_ids)]
    val_df = df[df[ID_COL].isin(val_ids)]
    test_df = df[df[ID_COL].isin(test_ids)]

    tv_dup = train_df.merge(val_df, on=key_cols, how="inner")
    tt_dup = train_df.merge(test_df, on=key_cols, how="inner")
    vt_dup = val_df.merge(test_df, on=key_cols, how="inner")

    print(f"Exact duplicate rows train↔val : {len(tv_dup)} {'-- LEAK' if len(tv_dup) else '-- clean'}")
    print(f"Exact duplicate rows train↔test: {len(tt_dup)} {'-- LEAK' if len(tt_dup) else '-- clean'}")
    print(f"Exact duplicate rows val↔test  : {len(vt_dup)} {'-- LEAK' if len(vt_dup) else '-- clean'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="path to patient_split_*.json")
    ap.add_argument("--data", required=True, help="path to the dataset CSV")
    args = ap.parse_args()

    train_ids, val_ids, test_ids = load_split(args.split)
    clean = step1_patient_overlap(train_ids, val_ids, test_ids)
    step1b_window_build_order(args.data, train_ids, val_ids, test_ids)
    step5_duplicate_rows(args.data, train_ids, val_ids, test_ids)

    print()
    print("=" * 70)
    print("NEXT: if all clean here, leak is NOT at split/row level.")
    print("      Move to Step 3 (label horizon leak) and Step 4 (val/test eval")
    print("      code-path mismatch) manually -- these need reading the label-")
    print("      building function and the two eval scripts side by side, not")
    print("      just data checks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
