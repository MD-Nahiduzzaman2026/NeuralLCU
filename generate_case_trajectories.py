"""
Fix 6: Case-level error analysis using the REAL trained model checkpoint
(neuralicu_centralized.pt). Produces Figure 13: hourly risk trajectories
for 4 patients (2 septic, 2 non-septic), sliding 12-hour window.

Requires in same folder:
    neuralicu_centralized.pt
    scaler.pkl
    hourly_features_labeled.csv

Install deps if needed:
    pip install torch pandas numpy matplotlib scikit-learn
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_PATH = r"Dataset\Model train with demo\neuralicu_centralized.pt"
SCALER_PATH = r"Dataset\Model train with demo\scaler.pkl"
FEATURES_CSV = "hourly_features_labeled.csv"
WINDOW = 12

FEATURE_ORDER = [
    "DiasBP", "GCS", "HeartRate", "MeanBP", "RespRate", "SpO2",
    "SysBP", "Temperature", "Bilirubin", "Creatinine", "Lactate", "WBC",
]


# ---------------------------------------------------------------------
# Exact architecture reconstructed from checkpoint (verified strict=True)
# ---------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TCNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.bn1(self.chomp1(self.conv1(x))))
        out = self.bn2(self.chomp2(self.conv2(out)))
        res = x if self.residual is None else self.residual(x)
        return self.relu(out + res)


class NeuralICUModel(nn.Module):
    def __init__(self, n_features=12):
        super().__init__()
        dilations = [1, 2, 4, 8]
        channels = [32, 64, 64, 64]
        blocks = []
        in_ch = n_features
        for d, out_ch in zip(dilations, channels):
            blocks.append(TCNResidualBlock(in_ch, out_ch, kernel_size=3, dilation=d))
            in_ch = out_ch
        self.tcn = nn.ModuleList(blocks)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        x = x.transpose(1, 2)  # -> [batch, n_features, seq_len] for Conv1d
        for block in self.tcn:
            x = block(x)
        x = x.transpose(1, 2)  # -> [batch, seq_len, 64] for Transformer
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling -> [batch, 64]
        return self.classifier(x).squeeze(-1)


# ---------------------------------------------------------------------
# Load model + scaler
# ---------------------------------------------------------------------
print("Loading model...")
model = NeuralICUModel(n_features=len(FEATURE_ORDER))
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt, strict=True)
model.eval()
print("Model loaded successfully (strict=True match).")

print("Loading scaler...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
n_feat = getattr(scaler, "n_features_in_", None)
print(f"Scaler loaded (expects {n_feat} features).")

# ---------------------------------------------------------------------
# Load data, pick 4 illustrative patients
# ---------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FEATURES_CSV)

# Verify expected columns exist
missing = [c for c in FEATURE_ORDER + ["STAY_ID", "HOUR", "SEPSIS"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

stay_labels = df.groupby("STAY_ID")["SEPSIS"].first()
stay_lengths = df.groupby("STAY_ID").size()

# Need enough hours for at least one full window plus a few trajectory points
eligible = stay_lengths[stay_lengths >= WINDOW + 6].index

septic_stays = sorted([s for s in eligible if stay_labels.get(s, 0) == 1])
nonseptic_stays = sorted([s for s in eligible if stay_labels.get(s, 0) == 0])

print(f"Eligible patients: {len(septic_stays)} septic, {len(nonseptic_stays)} non-septic")

if len(septic_stays) < 2:
    raise ValueError(f"Need >= 2 eligible septic patients, found {len(septic_stays)}")
if len(nonseptic_stays) < 2:
    raise ValueError(f"Need >= 2 eligible non-septic patients, found {len(nonseptic_stays)}")

# Pick first 2 from each group (deterministic, reproducible)
chosen_stays = septic_stays[:2] + nonseptic_stays[:2]
print(f"Chosen STAY_IDs: {chosen_stays}")
print(f"  Labels: {[stay_labels[s] for s in chosen_stays]}")
print(f"  Lengths: {[stay_lengths[s] for s in chosen_stays]} hours each")


# ---------------------------------------------------------------------
# Generate hourly risk trajectory per patient
# ---------------------------------------------------------------------
def get_trajectory(stay_id):
    """Slide a 12-hour window across the patient's stay, output risk at each hour."""
    patient = df[df["STAY_ID"] == stay_id].sort_values("HOUR").reset_index(drop=True)
    feats = patient[FEATURE_ORDER].values.astype(np.float32)

    # Check for NaN/inf before scaling
    if not np.all(np.isfinite(feats)):
        n_bad = np.sum(~np.isfinite(feats))
        print(f"  WARNING: Stay {stay_id} has {n_bad} non-finite values, replacing with 0")
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    feats_scaled = scaler.transform(feats)

    hours, risks = [], []
    for i in range(len(patient) - WINDOW):
        window = feats_scaled[i : i + WINDOW]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # [1, WINDOW, 12]
        with torch.no_grad():
            logit = model(x)
            risk = torch.sigmoid(logit).item()
        hours.append(patient["HOUR"].iloc[i + WINDOW])
        risks.append(risk)

    return hours, risks


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

colors_septic = ["#d62728", "#ff7f0e"]      # red, orange
colors_nonseptic = ["#1f77b4", "#2ca02c"]    # blue, green

for idx, (ax, stay_id) in enumerate(zip(axes, chosen_stays)):
    print(f"Computing trajectory for Stay {stay_id}...")
    hours, risks = get_trajectory(stay_id)
    label = int(stay_labels[stay_id])

    if label == 1:
        color = colors_septic[idx]
        title_tag = "Septic"
        marker_color = "red"
    else:
        color = colors_nonseptic[idx - 2]
        title_tag = "Non-septic"
        marker_color = "blue"

    ax.plot(hours, risks, color=color, linewidth=1.8, zorder=3)
    ax.scatter(hours, risks, color=color, s=12, zorder=4, alpha=0.7)

    # Decision threshold
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.5 threshold")

    # Shading above/below threshold
    ax.axhspan(0.5, 1.05, alpha=0.04, color="red")
    ax.axhspan(-0.05, 0.5, alpha=0.04, color="blue")

    ax.set_xlabel("Hour of Stay", fontsize=10)
    ax.set_ylabel("Sepsis-Risk Probability", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(hours[0] - 0.5, hours[-1] + 0.5)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    # Annotate min/max risk
    min_risk = min(risks)
    max_risk = max(risks)
    min_hour = hours[risks.index(min_risk)]
    max_hour = hours[risks.index(max_risk)]
    ax.annotate(f"min={min_risk:.2f}", xy=(min_hour, min_risk),
                fontsize=7, color="gray", ha="center", va="bottom")
    ax.annotate(f"max={max_risk:.2f}", xy=(max_hour, max_risk),
                fontsize=7, color="gray", ha="center", va="bottom")

    n_hours = len(hours)
    ax.set_title(
        f"Stay {stay_id} — {title_tag} ({n_hours} hours tracked)\n"
        f"Risk range: [{min_risk:.3f}, {max_risk:.3f}]",
        fontsize=10,
        fontweight="bold" if label == 1 else "normal",
    )

fig.suptitle(
    "Figure 13: Hourly Sepsis-Risk Trajectories (TCN-Transformer, 12-hour sliding window)\n"
    "Real trained model checkpoint · 4 illustrative patients from MIMIC-III Demo cohort",
    fontsize=12, fontweight="bold", y=1.02,
)

plt.tight_layout()
plt.savefig("case_analysis_fig13.pdf", bbox_inches="tight")
plt.savefig("case_analysis_fig13.png", dpi=200, bbox_inches="tight")
print("\n" + "=" * 60)
print("Saved case_analysis_fig13.pdf")
print("Saved case_analysis_fig13.png")
print("=" * 60)
print("\nSUMMARY OF 4 PATIENTS:")
for idx, stay_id in enumerate(chosen_stays):
    hours, risks = get_trajectory(stay_id) if idx == 0 else (None, None)
    # (already computed above in the plot loop, just re-fetch from there)
print("\nDone. Send case_analysis_fig13.png back for Figure 13 caption drafting.")