"""
Shared helpers for the revision experiments. Reads the SAME cache files that
train_eval_physionet2019.py (the cached/batched version) writes, so every
experiment here uses the frozen split and the identical normalized tensors.

Needs, in the same folder: config_physionet2019.py, preprocessing_dataset_csv.py
and outputs/patient_split_cache.json, normalized_tensors_cache.npz,
normalization_stats_physionet2019.json (all created by the training script).
"""

import json
import os

import numpy as np
import pandas as pd

from config_physionet2019 import DATA, LABEL_CONFIG, RAW_CHANNELS

OUT = DATA["output_dir"]
SPLIT_CACHE_PATH = f"{OUT}/patient_split_cache.json"
TENSOR_CACHE_PATH = f"{OUT}/normalized_tensors_cache.npz"
NORM_STATS_PATH = f"{OUT}/normalization_stats_physionet2019.json"
PRED_DIR = f"{OUT}/predictions"

WINDOW = LABEL_CONFIG["input_window_hours"]          # 12
HORIZON = LABEL_CONFIG["prediction_horizon_hours"]   # 6
LABEL_COL = LABEL_CONFIG["sepsis_label_column"]
N_CH = len(RAW_CHANNELS)                             # 12 channels x 3 features


def load_cache():
    cached = np.load(TENSOR_CACHE_PATH)
    return {k: cached[k] for k in cached.files}


def load_split():
    with open(SPLIT_CACHE_PATH) as f:
        return json.load(f)


def build_index(df: pd.DataFrame, patient_ids, y_cached=None, name=""):
    """
    Rebuilds, for one split, the per-window bookkeeping the cache does not
    store: patient id, prediction time, and the patient's sepsis onset hour.
    Window order is identical to the training script (patients ascending,
    windows ascending). If y_cached is given, the rebuilt labels must match it
    exactly; this guards against any ordering mistake.

    t_pred = index of the LAST observed hour of the window (start + 11).
    Onset  = first row with SepsisLabel == 1 (-1 if the patient never has one).
    """
    wanted = set(int(p) for p in patient_ids)
    pids, ts, ys, onsets = [], [], [], []
    for pid, g in df.groupby("Patient_ID"):
        if int(pid) not in wanted:
            continue
        labels = g[LABEL_COL].values
        n = len(labels)
        if n < WINDOW + 1:
            continue
        n_win = n - WINDOW
        starts = np.arange(n_win)
        cs = np.concatenate([[0], np.cumsum(labels == 1)])
        hi = np.minimum(starts + WINDOW + HORIZON, n)
        y = (cs[hi] - cs[starts + WINDOW]) > 0
        pos = np.flatnonzero(labels == 1)
        onset = int(pos[0]) if len(pos) else -1
        pids.append(np.full(n_win, int(pid)))
        ts.append(starts + WINDOW - 1)
        ys.append(y.astype(np.float32))
        onsets.append(np.full(n_win, onset))
    idx = {
        "pid": np.concatenate(pids),
        "t": np.concatenate(ts),
        "y": np.concatenate(ys),
        "onset": np.concatenate(onsets),
    }
    if y_cached is not None:
        if len(idx["y"]) != len(y_cached) or not np.array_equal(idx["y"], y_cached):
            raise RuntimeError(
                f"[{name}] rebuilt window labels do not match the cached labels "
                f"({len(idx['y'])} vs {len(y_cached)} windows). The patient/window "
                f"ordering assumption is wrong for this dataset; do not trust results."
            )
        print(f"[{name}] window index verified against cached labels ({len(y_cached)} windows)")
    return idx


def patient_slices(pid):
    """(start, end) slice per patient for a pid array that is grouped by patient."""
    change = np.flatnonzero(np.diff(pid)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(pid)]])
    return starts, ends


def load_indexed_splits():
    """Returns cache dict plus verified val/test indices."""
    from preprocessing_dataset_csv import load_dataset_csv
    cache = load_cache()
    split = load_split()
    df = load_dataset_csv()
    val_idx = build_index(df, split["val"], cache["y_val"], "val")
    test_idx = build_index(df, split["test"], cache["y_test"], "test")
    return cache, val_idx, test_idx


def ece_score(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
