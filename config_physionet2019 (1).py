"""
Alternate config: uses PhysioNet 2019 Sepsis Challenge data for
EVERY model (not just XGBoost). This needs no new credentialing —
you already have access to this dataset. It fixes Critical Issue #4
(cross-dataset baseline) by putting all 7 models on one dataset, and
helps Critical Issue #1 (instability) since n=40,336 vs n=100.

PhysioNet 2019 format: one .psv (pipe-separated) file per patient,
hourly rows, columns already include a per-hour SepsisLabel.
Download: https://physionet.org/content/challenge-2019/1.0.0/
(training_setA + training_setB, no CITI course required for this one).
"""

import os

DATA = {
    # Directory containing the unzipped training_setA/ and
    # training_setB/ .psv files from the 2019 Challenge download.
    "psv_dir": os.environ.get(
        "NEURALICU_PSV_DIR", "/home/claude/neuralicu_fix/data/physionet2019"
    ),
    "output_dir": "/home/claude/neuralicu_fix/outputs",
}

# ---------------------------------------------------------------------
# CHANNEL MAPPING — PhysioNet 2019 columns are NOT identical to the
# paper's original 14 MIMIC-derived channels. GCS and urine output are
# not present in this dataset. Mapped to 12 available channels; state
# this explicitly in the paper as a channel-availability limitation
# (ties into the existing graceful-degradation / LMIC argument).
# ---------------------------------------------------------------------
RAW_CHANNELS = [
    "HR",           # heart rate
    "O2Sat",        # SpO2
    "Temp",         # temperature
    "SBP",          # systolic BP
    "DBP",          # diastolic BP
    "MAP",          # mean arterial pressure
    "Resp",         # respiratory rate
    "WBC",          # white blood cell count
    "Lactate",      # serum lactate
    "Creatinine",   # creatinine
    "Platelets",    # platelet count
    "Bilirubin_total",  # total bilirubin
]
# Dropped vs. original 14: GCS, urine_output -- not present in this dataset.
DROPPED_CHANNELS_NOTE = (
    "GCS and urine output are not columns in the PhysioNet 2019 "
    "Challenge dataset and are excluded from this comparison. Report "
    "this explicitly as a dataset-availability limitation, not a "
    "silent change from the original 14-channel design."
)

LABEL_CONFIG = {
    "input_window_hours": 12,
    "prediction_horizon_hours": 6,
    "sepsis_label_column": "SepsisLabel",  # already provided per-hour by the Challenge
}

MODEL_LIST = [
    "logistic_regression",
    "xgboost",
    "random_forest",
    "gru_d",
    "tcn_only",
    "transformer_only",
    "tcn_transformer",
]

SEED_CONFIG = {
    "n_seeds": 25,
    "seeds": list(range(1000, 1025)),
    "bootstrap_resamples": 1000,
    "confidence_level": 0.95,
}

SPLIT_CONFIG = {
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "split_seed": 42,
    "split_manifest_path": "/home/claude/neuralicu_fix/outputs/patient_split_physionet2019.json",
}

TRAIN_CONFIG = {
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,   # can go larger — much more data than MIMIC Demo
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "early_stopping_metric": "val_auroc",
    "class_weighting": "inverse_frequency",
    "gradient_clip_norm": 1.0,
}
