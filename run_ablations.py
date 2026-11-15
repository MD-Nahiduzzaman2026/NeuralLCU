"""
Fix 4: Architecture ablation study on the SAME 100-patient MIMIC-III Demo
cohort (hourly_features_labeled.csv), matching Section 3.4's described
architecture: dilated causal TCN (dilations 1,2,4,8; kernel 3; 64 filters)
+ 4-head self-attention + global average pooling + 2-layer MLP head.

Given the instability we already measured in the XGBoost baseline (Fix 3),
every variant here is trained across MULTIPLE SEEDS and reported as
mean +/- std (range), not a single point estimate.

Install deps if needed:
    pip install torch pandas numpy scikit-learn --break-system-packages

Runtime: CPU is fine at this data scale. Expect ~5-15 min total for all
variants x all seeds (4 variants x 5 seeds = 20 short training runs).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)

FEATURES_CSV = "hourly_features_labeled.csv"
SEEDS = [0, 1, 7, 42, 123]
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3

df = pd.read_csv(FEATURES_CSV)
feature_cols = [c for c in df.columns if c not in ("STAY_ID", "HOUR", "SEPSIS", "SUBJECT_ID")]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}\n")


# ---------------------------------------------------------------------
# Data: build (sequence, label, subject_id) windows -- NOT flattened,
# since these models need [timesteps, features] sequences.
# ---------------------------------------------------------------------
def build_windows(df_subset, window):
    X, y, subj = [], [], []
    for sid in df_subset["STAY_ID"].unique():
        patient = df_subset[df_subset["STAY_ID"] == sid].sort_values("HOUR").reset_index(drop=True)
        if len(patient) < window + 1:
            continue
        feats = patient[feature_cols].values.astype(np.float32)
        labels = patient["SEPSIS"].values
        subject_id = patient["SUBJECT_ID"].iloc[0]
        for i in range(len(patient) - window):
            X.append(feats[i:i + window])
            y.append(labels[i + window])
            subj.append(subject_id)
    return np.array(X), np.array(y, dtype=np.float32), np.array(subj)


def make_split(seed, window):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    X_train, y_train, _ = build_windows(df_train, window)
    X_test, y_test, subj_test = build_windows(df_test, window)

    # Fit scaler on train only, apply to both (avoid test leakage into scaling)
    scaler = StandardScaler()
    n_feat = X_train.shape[2]
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    return X_train, y_train, X_test, y_test, subj_test


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class Chomp1d(nn.Module):
    """Trims right-side padding so causal convolutions never see the future."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp = Chomp1d(pad)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.chomp(self.conv(x)))


class TCNTransformer(nn.Module):
    """Full architecture: dilated causal TCN + optional 4-head self-attention."""
    def __init__(self, n_features, use_attention=True, tcn_channels=64):
        super().__init__()
        dilations = [1, 2, 4, 8]
        layers = []
        in_ch = n_features
        for d in dilations:
            layers.append(TCNBlock(in_ch, tcn_channels, kernel_size=3, dilation=d))
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*layers)

        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=tcn_channels, num_heads=4, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(tcn_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [batch, time, features] -> conv wants [batch, features, time]
        x = x.transpose(1, 2)
        x = self.tcn(x)                 # [batch, channels, time]
        x = x.transpose(1, 2)           # [batch, time, channels]
        if self.use_attention:
            x, _ = self.attention(x, x, x)
        x = x.mean(dim=1)               # global average pooling over time
        return self.head(x).squeeze(-1)  # logits


class LSTMBaseline(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------
# Train + evaluate one (variant, seed) combination
# ---------------------------------------------------------------------
def train_and_eval(model_fn, seed, window):
    X_train, y_train, X_test, y_test, subj_test = make_split(seed, window)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None  # degenerate split, skip

    torch.manual_seed(seed)
    model = model_fn(n_features=X_train.shape[2])

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)

    n = len(X_train_t)
    for epoch in range(EPOCHS):
        model.train()  # <-- critical: dropout/attention active during training
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_train_t[idx])
            loss = criterion(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()  # <-- critical: this is the exact bug we flagged earlier in
                  # the original training curve; must be set before evaluating
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.sigmoid(test_logits).numpy()

    auroc = roc_auc_score(y_test, test_probs)
    return auroc


# ---------------------------------------------------------------------
# Run all variants across all seeds
# ---------------------------------------------------------------------
variants = {
    "Full TCN-Transformer (12hr)": (lambda n_features: TCNTransformer(n_features, use_attention=True), 12),
    "TCN-only, no attention (12hr)": (lambda n_features: TCNTransformer(n_features, use_attention=False), 12),
    "LSTM baseline (12hr)": (lambda n_features: LSTMBaseline(n_features), 12),
    "Full TCN-Transformer (6hr window)": (lambda n_features: TCNTransformer(n_features, use_attention=True), 6),
    "Full TCN-Transformer (24hr window)": (lambda n_features: TCNTransformer(n_features, use_attention=True), 24),
}

results = {}
for name, (model_fn, window) in variants.items():
    print(f"=== {name} ===")
    scores = []
    for seed in SEEDS:
        auroc = train_and_eval(model_fn, seed, window)
        if auroc is not None:
            print(f"  seed {seed}: AUROC = {auroc:.4f}")
            scores.append(auroc)
        else:
            print(f"  seed {seed}: skipped (degenerate split)")
    if scores:
        results[name] = scores
        print(f"  --> mean {np.mean(scores):.4f}, std {np.std(scores):.4f}, "
              f"range {min(scores):.4f}-{max(scores):.4f}\n")
    else:
        print("  --> no valid runs\n")

print("\n=== SUMMARY TABLE (paste into Table 6) ===")
print(f"{'Variant':<38} {'Mean AUROC':<12} {'Std':<8} {'Range'}")
for name, scores in results.items():
    print(f"{name:<38} {np.mean(scores):<12.4f} {np.std(scores):<8.4f} "
          f"{min(scores):.4f}-{max(scores):.4f}")
