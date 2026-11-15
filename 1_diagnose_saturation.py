"""
STEP 1: DIAGNOSE SATURATION
============================
What this does:
    Checks whether your model's predicted probabilities are stuck at
    0.0 / 1.0 (saturated) instead of spreading out honestly between
    0 and 1. Produces a calibration curve image and prints two numbers
    you should paste into your notes/paper.

How to run:
    python 1_diagnose_saturation.py

Before running, edit the CONFIG section below to point at your own
saved predictions file.

What you need:
    A file (CSV or .npy) with two columns/arrays:
      - true labels (0 or 1)
      - predicted probabilities (a number between 0 and 1)

    If you don't have this file yet, ask whoever ran the model
    training to save the validation-set predictions as a CSV with
    columns "y_true" and "y_prob".

Output:
    - Prints "Fraction of predictions saturated at 0/1" — if this is
      above roughly 0.3 (30%), that's a strong sign of the problem.
    - Prints the Brier score — lower is better; 0.25 or higher usually
      means the model isn't calibrated well.
    - Saves calibration_curve.png — if the blue line hugs the bottom
      then jumps straight to the top (instead of following the gray
      diagonal), that visually confirms saturation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ============================================================
# CONFIG — edit these two lines to match your file
# ============================================================
INPUT_CSV = "predictions.csv"   # must have columns: y_true, y_prob
OUTPUT_IMAGE = "calibration_curve.png"
# ============================================================


def load_predictions(csv_path):
    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        raise ValueError(
            f"'{csv_path}' must have columns named exactly 'y_true' and "
            f"'y_prob'. Found columns: {list(df.columns)}"
        )
    return df["y_true"].to_numpy(), df["y_prob"].to_numpy()


def diagnose_saturation(y_true, y_prob, output_image):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    frac_saturated = np.mean((y_prob < 0.01) | (y_prob > 0.99))
    print(f"Fraction of predictions saturated at 0/1: {frac_saturated:.3f}")
    if frac_saturated > 0.3:
        print("  -> This is HIGH. Predictions are likely saturated.")
    else:
        print("  -> This looks reasonable, not obviously saturated.")

    brier = brier_score_loss(y_true, y_prob)
    print(f"Brier score: {brier:.4f} (lower is better; 0.25 = uninformative baseline)")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.title("Calibration curve")
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration curve to: {output_image}")

    return frac_saturated, brier


if __name__ == "__main__":
    y_true, y_prob = load_predictions(INPUT_CSV)
    diagnose_saturation(y_true, y_prob, OUTPUT_IMAGE)
