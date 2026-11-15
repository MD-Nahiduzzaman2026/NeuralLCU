"""
STEP 4A: TEMPERATURE SCALING (QUICK FIX, NO RETRAINING)
==========================================================
What this does:
    Takes your ALREADY-TRAINED model and learns a single extra number
    (called "temperature") that softens overconfident 0/1 predictions
    into honest probabilities. This does NOT change the model's
    weights or its ranking of who's high-risk vs low-risk — it only
    fixes how confident the numbers look. It's the cheapest fix to
    try first.

    Only use this if Step 3 did NOT find an obvious leaking feature.
    If Step 3 found a leaking static feature, fix that first (drop
    the feature and retrain) — temperature scaling on a model that's
    cheating just makes the cheating look more honest, it doesn't
    remove it.

How to run:
    python 4_temperature_scaling_fix.py

Before running, fill in the "LOAD YOUR OWN" section near the bottom.

What you need:
    - Your trained PyTorch model
    - A validation set DataLoader (data the model wasn't trained on)

Output:
    Prints the learned temperature value, and saves a new file
    "temperature_value.txt" with that number in it. To use it later,
    divide your model's raw output (logits) by this number before
    applying sigmoid — see the note at the bottom of this file for
    how to apply it at prediction time.
"""

import torch

# ============================================================
# CONFIG
# ============================================================
OUTPUT_TEMPERATURE_FILE = "temperature_value.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================


class TemperatureScaler(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature


def fit_temperature(model, val_loader, device):
    scaler = TemperatureScaler(model).to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = torch.nn.BCEWithLogitsLoss()

    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            logits_list.append(model(x.to(device)))
            labels_list.append(y.to(device))
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / scaler.temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    learned_temp = scaler.temperature.item()
    print(f"Learned temperature: {learned_temp:.3f}")

    if learned_temp > 1.0:
        print("  -> Temperature > 1 means the model WAS overconfident. This fix should help.")
    else:
        print("  -> Temperature <= 1 means the model wasn't really overconfident.")
        print("     The saturation is more likely coming from a real static-feature")
        print("     shortcut (see step 3) rather than plain overconfidence.")

    return scaler, learned_temp


if __name__ == "__main__":
    # -------- LOAD YOUR OWN: model and validation data --------
    # Replace this with your actual model loading and DataLoader setup.
    #
    # model = torch.load("your_model.pt", map_location=DEVICE)
    # val_loader = ...  # a torch DataLoader over your validation set
    #
    # Uncomment the two lines above (with your real paths/objects) and
    # delete the next line before running.
    raise NotImplementedError(
        "Fill in the 'LOAD YOUR OWN' section above with your real model "
        "and validation DataLoader, then remove this line."
    )

    scaler, learned_temp = fit_temperature(model, val_loader, DEVICE)

    with open(OUTPUT_TEMPERATURE_FILE, "w") as f:
        f.write(str(learned_temp))
    print(f"Saved temperature value to: {OUTPUT_TEMPERATURE_FILE}")

    print("\nTo use this at prediction time:")
    print("  1. Run your model to get raw logits (output before sigmoid)")
    print("  2. Divide the logits by the learned temperature")
    print("  3. Then apply sigmoid to get the final, better-calibrated probability")
