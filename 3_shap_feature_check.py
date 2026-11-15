"""
STEP 3: FIND THE LEAKING FEATURE (IF ANY)
============================================
What this does:
    Ranks which input features your model relies on most. If the #1
    feature is something that never changes during a patient's stay
    (like admission type, hospital ID, or insurance code), that's a
    strong sign the model is "cheating" by memorizing patient type
    instead of learning from real-time vital sign trends. That
    shortcut is a common cause of saturated 0/1 predictions.

How to run:
    python 3_shap_feature_check.py

Before running, edit the CONFIG section below. You (or whoever
trained the model) will need to fill in the two "LOAD YOUR OWN"
sections near the bottom, since this needs your actual trained
model and data loaded in memory — this part cannot be fully
generic like the other two scripts.

What you need:
    - Your trained PyTorch model file
    - A saved sample of input data (as a PyTorch tensor) to explain
    - The list of feature names, in the same order as the model's
      input columns

Output:
    Prints the top 10 most important features, ranked. Look at #1:
      - If it's a vital sign or lab value that changes over time
        (e.g. heart rate, lactate) -> likely fine, not leaking.
      - If it's a static/administrative field -> that's the fix target
        for step 4 (drop it and retrain).
"""

import numpy as np
import torch

try:
    import shap
except ImportError:
    raise ImportError(
        "This script needs the 'shap' package. Install it first with:\n"
        "    pip install shap --break-system-packages"
    )

# ============================================================
# CONFIG — edit these to match your project
# ============================================================
MODEL_PATH = "your_model.pt"          # path to your saved PyTorch model
DATA_PATH = "sample_input.pt"         # path to a saved tensor of sample inputs
FEATURE_NAMES_PATH = "feature_names.txt"  # one feature name per line, in column order
NUM_BACKGROUND_SAMPLES = 100          # how many samples SHAP uses as a baseline
NUM_EXPLAIN_SAMPLES = 200             # how many samples to actually explain
# ============================================================


def load_feature_names(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def check_static_leakage(model, X_sample, feature_names):
    model.eval()
    explainer = shap.DeepExplainer(model, X_sample[:NUM_BACKGROUND_SAMPLES])
    shap_values = explainer.shap_values(X_sample[:NUM_EXPLAIN_SAMPLES])

    # Average absolute importance across samples and time steps
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
    ranked = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])

    print("Top 10 features by importance:")
    for name, val in ranked[:10]:
        print(f"  {name}: {val:.4f}")

    return ranked


if __name__ == "__main__":
    # -------- LOAD YOUR OWN: model --------
    # Replace this with however your model class is normally loaded.
    # Example if you saved the whole model object:
    model = torch.load(MODEL_PATH, map_location="cpu")
    # Example if you saved only weights, uncomment and adjust instead:
    # model = YourModelClass(...)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # -------- LOAD YOUR OWN: sample input data --------
    # This should be a torch.Tensor shaped like your model's real input,
    # e.g. (num_samples, sequence_length, num_features)
    X_sample = torch.load(DATA_PATH, map_location="cpu")

    feature_names = load_feature_names(FEATURE_NAMES_PATH)

    check_static_leakage(model, X_sample, feature_names)
