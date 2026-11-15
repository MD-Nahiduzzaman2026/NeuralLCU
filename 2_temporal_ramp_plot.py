"""
STEP 2: CHECK TEMPORAL RAMP-UP
================================
What this does:
    For a handful of patients who actually had the event (e.g. sepsis
    onset), plots their predicted risk over time. A healthy "early
    warning" model should show risk climbing gradually as the event
    approaches. If the line is flat near 0 and then jumps straight to
    1 right at (or after) the event, that's the saturation problem.

How to run:
    python 2_temporal_ramp_plot.py

Before running, edit the CONFIG section below.

What you need:
    A CSV with one row per (patient, timestamp) prediction, with columns:
      - patient_id
      - time            (a number: hours since admission, or similar)
      - y_prob          (predicted probability at that time)
      - event_time      (the time the event happened for that patient;
                          same value repeated for all rows of that patient)

    Ask whoever ran the model to export per-timestep predictions for
    a set of true-positive patients (patients who actually had the event).

Output:
    One PNG image per patient, e.g. ramp_patient_1042.png, saved into
    the same folder. Look at these images directly — no numbers to read,
    just look at the shape of the line.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — edit this to match your file
# ============================================================
INPUT_CSV = "temporal_predictions.csv"   # columns: patient_id, time, y_prob, event_time
MAX_PATIENTS_TO_PLOT = 15
# ============================================================


def plot_temporal_ramp(patient_id, timestamps, probs, event_time):
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, probs, marker="o")
    plt.axvline(event_time, color="red", linestyle="--", label="Event onset")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Time")
    plt.ylabel("Predicted probability")
    plt.title(f"Patient {patient_id} risk trajectory")
    plt.legend()
    filename = f"ramp_patient_{patient_id}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    required_cols = {"patient_id", "time", "y_prob", "event_time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    patient_ids = df["patient_id"].unique()[:MAX_PATIENTS_TO_PLOT]
    print(f"Plotting {len(patient_ids)} patients...")

    for pid in patient_ids:
        sub = df[df["patient_id"] == pid].sort_values("time")
        plot_temporal_ramp(
            patient_id=pid,
            timestamps=sub["time"].to_numpy(),
            probs=sub["y_prob"].to_numpy(),
            event_time=sub["event_time"].iloc[0],
        )

    print("\nDone. Open the saved PNG files and look at the shape of each line:")
    print("  - Gradual rise toward the red line = good, real early warning")
    print("  - Flat near 0 then instant jump to 1 = saturation problem")
