"""
Fix 8: Differential privacy utility tradeoff, measured on the same 100-patient
MIMIC-III Demo cohort.

IMPORTANT SUBSTITUTION: this uses the LSTM baseline architecture, not the full
TCN-Transformer with attention. Opacus (the DP library your Section 3.6 already
cites -- Abadi et al. via Meta AI's Opacus) computes per-sample gradients via
functorch, which does not yet support nn.MultiheadAttention. Implementing
custom per-sample gradients for attention is possible but substantial extra
work; using the LSTM baseline here is the standard practical workaround and
should be stated as such in the paper rather than silently substituted.

Targets the paper's stated privacy budget: epsilon=1.0, delta=1e-5,
gradient clipping norm C=1.0 (Section 3.6).

Install deps if needed:
    pip install torch opacus pandas numpy scikit-learn --break-system-packages
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from opacus import PrivacyEngine
from opacus.layers import DPLSTM

FEATURES_CSV = "hourly_features_labeled.csv"
SEEDS = [0, 1, 7, 42, 123]
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
WINDOW = 12

TARGET_EPSILON = 1.0
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0  # matches Section 3.6's C = 1.0

df = pd.read_csv(FEATURES_CSV)
feature_cols = [c for c in df.columns if c not in ("STAY_ID", "HOUR", "SEPSIS", "SUBJECT_ID")]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}\n")


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


def make_split(seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["SUBJECT_ID"]))
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    X_train, y_train, _ = build_windows(df_train, WINDOW)
    X_test, y_test, _ = build_windows(df_test, WINDOW)

    scaler = StandardScaler()
    n_feat = X_train.shape[2]
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    return X_train, y_train, X_test, y_test


class LSTMBaseline(nn.Module):
    """Uses DPLSTM (Opacus-compatible drop-in for nn.LSTM) so per-sample
    gradients can be computed for the private training run."""
    def __init__(self, n_features, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = DPLSTM(n_features, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_eval(seed, use_dp):
    X_train, y_train, X_test, y_test = make_split(seed)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None, None

    torch.manual_seed(seed)
    model = LSTMBaseline(n_features=X_train.shape[2])

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    achieved_epsilon = None
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=TARGET_EPSILON,
            target_delta=TARGET_DELTA,
            epochs=EPOCHS,
            max_grad_norm=MAX_GRAD_NORM,
        )

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    if use_dp:
        achieved_epsilon = privacy_engine.get_epsilon(TARGET_DELTA)

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test))
        test_probs = torch.sigmoid(test_logits).numpy()

    auroc = roc_auc_score(y_test, test_probs)
    return auroc, achieved_epsilon


print("=== WITHOUT DP ===")
no_dp_scores = []
for seed in SEEDS:
    auroc, _ = train_eval(seed, use_dp=False)
    if auroc is not None:
        print(f"  seed {seed}: AUROC = {auroc:.4f}")
        no_dp_scores.append(auroc)

print("\n=== WITH DP (target eps=1.0, delta=1e-5, C=1.0) ===")
dp_scores = []
epsilons = []
for seed in SEEDS:
    auroc, eps = train_eval(seed, use_dp=True)
    if auroc is not None:
        print(f"  seed {seed}: AUROC = {auroc:.4f}, achieved epsilon = {eps:.4f}")
        dp_scores.append(auroc)
        epsilons.append(eps)

print("\n=== SUMMARY (paste into Section 3.6 / paper text) ===")
if no_dp_scores:
    print(f"Without DP: mean AUROC {np.mean(no_dp_scores):.4f}, std {np.std(no_dp_scores):.4f}, "
          f"range {min(no_dp_scores):.4f}-{max(no_dp_scores):.4f}")
if dp_scores:
    print(f"With DP:    mean AUROC {np.mean(dp_scores):.4f}, std {np.std(dp_scores):.4f}, "
          f"range {min(dp_scores):.4f}-{max(dp_scores):.4f}")
    print(f"Achieved epsilon (mean across seeds): {np.mean(epsilons):.4f} (target was {TARGET_EPSILON})")
if no_dp_scores and dp_scores:
    print(f"Mean performance drop: {np.mean(no_dp_scores) - np.mean(dp_scores):.4f}")
