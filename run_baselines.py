import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import os

print("=== NeuralICU Baseline Data Extractor ===")
print("Looking for data...")

# Load the data
file_path = 'training_setA.csv'
if not os.path.exists(file_path):
    print(f"ERROR: Could not find {file_path}. Please ensure this script is in the same folder as the extracted data.")
    exit()

df = pd.read_csv(file_path)
print(f"Loaded {len(df)} rows of data.")

# Select the 6 most critical features from your literature review (that are available in this dataset)
features = ['HR', 'MAP', 'Resp', 'Lactate', 'Age', 'HospAdmTime', 'ICULOS']
label = 'SepsisLabel'

# Drop rows with missing critical features for a clean baseline run
df = df.dropna(subset=features + [label])

# Patient-wise split (CRITICAL: prevents data leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['Patient_ID']))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

X_train = train_df[features]
y_train = train_df[label]
X_test = test_df[features]
y_test = test_df[label]

print(f"Training on {len(train_df)} rows, Testing on {len(test_df)} rows.")
print("Training XGBoost model... (This takes 1-3 minutes)")

# Train XGBoost (handles the skewed sepsis data well)
model = xgb.XGBClassifier(
    eval_metric='aucpr', 
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), # Fixes class imbalance
    max_depth=4, 
    learning_rate=0.1, 
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Predict probabilities
y_prob = model.predict_proba(X_test[features])[:, 1]

# Calculate Metrics
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr = average_precision_score(y_test, y_prob)

# Calculate 95% Confidence Interval for AUROC using Bootstrapping
print("Calculating 95% Confidence Intervals...")
n_bootstraps = 1000
bootstrapped_scores = []
rng = np.random.RandomState(42)

for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test.iloc[indices])) < 2:
        continue
    score = roc_auc_score(y_test.iloc[indices], y_prob[indices])
    bootstrapped_scores.append(score)

sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()
ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]

# Create output folder
if not os.path.exists('results'):
    os.makedirs('results')

# Save Numbers to a text file
with open('results/model_metrics.txt', 'w') as f:
    f.write(f"--- COPY THESE NUMBERS INTO YOUR PAPER ---\n")
    f.write(f"Dataset: PhysioNet 2019 (Derived from MIMIC-III)\n")
    f.write(f"Test Set Size: {len(test_df)} hours\n")
    f.write(f"XGBoost Baseline AUROC: {auc_roc:.3f}\n")
    f.write(f"95% Confidence Interval: {ci_lower:.3f} - {ci_upper:.3f}\n")
    f.write(f"XGBoost Baseline AUPRC: {auc_pr:.3f}\n")
    
print("\n=== SUCCESS ===")
print(f"XGBoost Baseline AUROC: {auc_roc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
print(f"XGBoost Baseline AUPRC: {auc_pr:.3f}")
print("Numbers saved to 'results/model_metrics.txt'")

# --- FIGURE 1: ROC CURVE ---
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#4C72B0', lw=2, label=f'XGBoost Baseline (AUROC = {auc_roc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=300)
print("Saved: results/roc_curve.png")

# --- FIGURE 2: CALIBRATION PLOT ---
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', color='#DD8452', lw=2, label='XGBoost Baseline')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title('Calibration Curve (Reliability Diagram)', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/calibration_curve.png', dpi=300)
print("Saved: results/calibration_curve.png")
plt.close()