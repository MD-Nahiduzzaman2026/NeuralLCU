"""
build_master_results_table.py

Consolidates every FINAL, CORRECTED result from this project into one
master table: reads the PhysioNet 2019 CSVs directly (already correct,
saved by stats_analysis_physionet2019.py), and PARSES the pooled-k-fold
log files for MIMIC-III ablations, XGBoost, and the DP tradeoff -- the
corrected versions, not the earlier single-split logs that had the
prevalence-mismatch confound.

IMPORTANT: this script points at these SPECIFIC log files, which are the
final corrected ones from this project's debugging session. If you rerun
any step and it writes a differently-named log, update the paths below.

    PhysioNet 2019 summary:      outputs/model_summary_table_physionet2019.csv
    PhysioNet 2019 significance: outputs/significance_vs_proposed_physionet2019.csv
    MIMIC-III ablations:         outputs/logs/step3b_pooled.log
    MIMIC-III XGBoost:           outputs/logs/step3a_kfold.log
    MIMIC-III DP tradeoff:       outputs/logs/step4_kfold_v2.log

Output:
    outputs/master_results_table.csv   -- one row per (dataset, model)
    outputs/master_results_table.md    -- same table, Markdown, paste-ready

Run:
    python build_master_results_table.py
"""

import re
import os
import pandas as pd

OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

PHYSIONET_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "model_summary_table_physionet2019.csv")
PHYSIONET_SIG_CSV = os.path.join(OUTPUT_DIR, "significance_vs_proposed_physionet2019.csv")
MIMIC_ABLATIONS_LOG = os.path.join(LOG_DIR, "step3b_pooled.log")
MIMIC_XGBOOST_LOG = os.path.join(LOG_DIR, "step3a_kfold.log")
MIMIC_DP_LOG = os.path.join(LOG_DIR, "step4_kfold_v2.log")

rows = []  # each: dict with Dataset, Model, Mean_AUROC, Std_AUROC, N, Range, Notes


def read_text_smart(path):
    """Reads a text file regardless of whether it's UTF-8/ANSI (from cmd.exe's
    `>` redirection) or UTF-16 (PowerShell's Tee-Object -FilePath writes UTF-16
    by default, which breaks naive UTF-8 decoding -- every character gets
    padded with a null byte, silently corrupting the text instead of raising
    an error). Detects encoding from the byte-order-mark when present, falls
    back to UTF-8, and as a last resort UTF-16 without a BOM."""
    with open(path, "rb") as f:
        raw = f.read()

    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def require_file(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {what} at '{path}'. Check the filename matches what "
            f"you actually saved this run, or update the path constants at the "
            f"top of this script."
        )


# ---------------------------------------------------------------------
# 1. PhysioNet 2019 (25 seeds, n=40,336 patients) -- read CSVs directly
# ---------------------------------------------------------------------
require_file(PHYSIONET_SUMMARY_CSV, "PhysioNet 2019 summary CSV")
physionet_summary = pd.read_csv(PHYSIONET_SUMMARY_CSV)

for _, r in physionet_summary.iterrows():
    rows.append({
        "Dataset": "PhysioNet 2019 (n=40,336)",
        "Model": r["model"],
        "Mean_AUROC": round(r["mean_auroc"], 4),
        "Std_AUROC": round(r["std_auroc"], 4),
        "N_runs": int(r["n_seeds"]),
        "Range": f"{r['min_auroc']:.4f}-{r['max_auroc']:.4f}",
        "Notes": "single random 80/20 split per seed (large-n dataset; "
                 "prevalence-mismatch confound not applicable at this scale)",
    })

physionet_sig = None
if os.path.exists(PHYSIONET_SIG_CSV):
    physionet_sig = pd.read_csv(PHYSIONET_SIG_CSV)


# ---------------------------------------------------------------------
# 2. MIMIC-III ablations -- parse the pooled-k-fold summary table
# ---------------------------------------------------------------------
require_file(MIMIC_ABLATIONS_LOG, "MIMIC-III ablations pooled k-fold log")
ablations_text = read_text_smart(MIMIC_ABLATIONS_LOG)

ablation_row_pattern = re.compile(
    r"^(?P<model>.+?)\s{2,}"
    r"(?P<mean>\d+\.\d+)\s+"
    r"(?P<std>\d+\.\d+)\s+"
    r"(?P<n>\d+)\s+"
    r"(?P<range>\d+\.\d+-\d+\.\d+)\s*$",
    re.MULTILINE,
)

summary_section_match = re.search(
    r"SUMMARY TABLE \(pooled out-of-fold AUROC per repeat.*?\)\s*===\s*\n(.*)",
    ablations_text, re.DOTALL,
)
if not summary_section_match:
    raise ValueError(
        f"Could not find the pooled-AUROC summary table in {MIMIC_ABLATIONS_LOG}. "
        f"Is this the corrected pooled-OOF log (from the FIXED run_ablations_kfold.py), "
        f"not an earlier per-fold-only version?"
    )
summary_section = summary_section_match.group(1)

n_ablation_rows = 0
for match in ablation_row_pattern.finditer(summary_section):
    model_name = match.group("model").strip()
    if model_name.lower().startswith("variant"):
        continue
    rows.append({
        "Dataset": "MIMIC-III Demo (n=99, stratified k-fold)",
        "Model": model_name,
        "Mean_AUROC": round(float(match.group("mean")), 4),
        "Std_AUROC": round(float(match.group("std")), 4),
        "N_runs": int(match.group("n")),
        "Range": match.group("range"),
        "Notes": "pooled out-of-fold AUROC, 5 repeats x 5-fold patient-level "
                 "stratified CV; near chance (0.50) across all variants",
    })
    n_ablation_rows += 1

if n_ablation_rows == 0:
    raise ValueError(
        f"Parsed {MIMIC_ABLATIONS_LOG} but extracted zero model rows -- the log "
        f"format may not match what this script expects. Check the file wasn't "
        f"truncated or altered."
    )


# ---------------------------------------------------------------------
# 3. MIMIC-III XGBoost -- parse the pooled-k-fold summary block
# ---------------------------------------------------------------------
require_file(MIMIC_XGBOOST_LOG, "MIMIC-III XGBoost pooled k-fold log")
xgb_text = read_text_smart(MIMIC_XGBOOST_LOG)

xgb_auroc_match = re.search(
    r"Mean AUROC:\s*(?P<mean>\d+\.\d+),\s*std:\s*(?P<std>\d+\.\d+),\s*"
    r"range:\s*(?P<range>\d+\.\d+-\d+\.\d+),\s*n_repeats=(?P<n>\d+)",
    xgb_text,
)
xgb_auprc_match = re.search(
    r"Mean AUPRC:\s*(?P<mean>\d+\.\d+),\s*std:\s*(?P<std>\d+\.\d+)",
    xgb_text,
)
if not xgb_auroc_match:
    raise ValueError(
        f"Could not find the pooled AUROC summary line in {MIMIC_XGBOOST_LOG}. "
        f"Is this the corrected pooled-OOF log (from xgboost_same_cohort_kfold.py)?"
    )

rows.append({
    "Dataset": "MIMIC-III Demo (n=99, stratified k-fold)",
    "Model": "xgboost",
    "Mean_AUROC": round(float(xgb_auroc_match.group("mean")), 4),
    "Std_AUROC": round(float(xgb_auroc_match.group("std")), 4),
    "N_runs": int(xgb_auroc_match.group("n")),
    "Range": xgb_auroc_match.group("range"),
    "Notes": (
        f"pooled OOF AUROC, 5 repeats x 5-fold; AUPRC "
        f"{xgb_auprc_match.group('mean') if xgb_auprc_match else 'N/A'} "
        f"(base rate ~26-30%, so near-chance); near chance (0.50)"
    ),
})


# ---------------------------------------------------------------------
# 4. MIMIC-III DP tradeoff -- parse without-DP / with-DP + Wilcoxon
# ---------------------------------------------------------------------
require_file(MIMIC_DP_LOG, "MIMIC-III DP tradeoff pooled k-fold log")
dp_text_raw = read_text_smart(MIMIC_DP_LOG)

dp_text = re.sub(r"\s+", " ", dp_text_raw)

no_dp_match = re.search(
    r"Without DP:\s*mean AUROC\s*(?P<mean>\d+\.\d+),\s*std\s*(?P<std>\d+\.\d+),\s*"
    r"range\s*(?P<range>\d+\.\d+-\d+\.\d+),\s*n_repeats=(?P<n>\d+)",
    dp_text,
)
dp_match = re.search(
    r"With DP:\s*mean AUROC\s*(?P<mean>\d+\.\d+),\s*std\s*(?P<std>\d+\.\d+),\s*"
    r"range\s*(?P<range>\d+\.\d+-\d+\.\d+),\s*n_repeats=(?P<n>\d+)",
    dp_text,
)
eps_match = re.search(r"Achieved epsilon.*?:\s*(?P<eps>\d+\.\d+)\s*\(target was (?P<target>[\d.]+)\)", dp_text)
wilcoxon_match = re.search(r"stat=(?P<stat>[\d.]+),\s*p=(?P<p>[\d.]+)", dp_text)

if not (no_dp_match and dp_match):
    dp_context_matches = [m.start() for m in re.finditer(r"DP", dp_text)]
    context_snippets = []
    for pos in dp_context_matches[:5]:
        context_snippets.append(dp_text[max(0, pos - 40):pos + 80])
    context_str = "\n  ...".join(context_snippets) if context_snippets else "(no 'DP' occurrences found at all)"
    raise ValueError(
        f"Could not find both 'Without DP' and 'With DP' summary lines in "
        f"{MIMIC_DP_LOG} even after whitespace normalization. Is this the "
        f"corrected pooled-OOF log (from run_dp_tradeoff_kfold.py)? "
        f"Nearby text found:\n  ...{context_str}"
    )

eps_note = f", target eps={eps_match.group('target')}, achieved eps={eps_match.group('eps')}" if eps_match else ""
wilcoxon_note = f", paired Wilcoxon p={wilcoxon_match.group('p')} (n=5, not significant)" if wilcoxon_match else ""

rows.append({
    "Dataset": "MIMIC-III Demo (n=99, stratified k-fold)",
    "Model": "LSTM baseline, without DP",
    "Mean_AUROC": round(float(no_dp_match.group("mean")), 4),
    "Std_AUROC": round(float(no_dp_match.group("std")), 4),
    "N_runs": int(no_dp_match.group("n")),
    "Range": no_dp_match.group("range"),
    "Notes": f"pooled OOF AUROC, 5 repeats x 5-fold{wilcoxon_note}",
})
rows.append({
    "Dataset": "MIMIC-III Demo (n=99, stratified k-fold)",
    "Model": "LSTM baseline, WITH DP (Opacus)",
    "Mean_AUROC": round(float(dp_match.group("mean")), 4),
    "Std_AUROC": round(float(dp_match.group("std")), 4),
    "N_runs": int(dp_match.group("n")),
    "Range": dp_match.group("range"),
    "Notes": f"pooled OOF AUROC, 5 repeats x 5-fold{eps_note}{wilcoxon_note}",
})


# ---------------------------------------------------------------------
# Assemble and save
# ---------------------------------------------------------------------
master_df = pd.DataFrame(rows)
master_df = master_df.sort_values(["Dataset", "Mean_AUROC"], ascending=[True, False]).reset_index(drop=True)

csv_path = os.path.join(OUTPUT_DIR, "master_results_table.csv")
master_df.to_csv(csv_path, index=False)

md_path = os.path.join(OUTPUT_DIR, "master_results_table.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Master Results Table\n\n")
    for dataset_name, group in master_df.groupby("Dataset", sort=False):
        f.write(f"## {dataset_name}\n\n")
        f.write("| Model | Mean AUROC | Std | N runs | Range | Notes |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in group.iterrows():
            f.write(f"| {r['Model']} | {r['Mean_AUROC']:.4f} | {r['Std_AUROC']:.4f} | "
                     f"{r['N_runs']} | {r['Range']} | {r['Notes']} |\n")
        f.write("\n")

print(master_df.to_string(index=False))
print(f"\nSaved -> {csv_path}")
print(f"Saved -> {md_path}")

if physionet_sig is not None:
    print("\n=== PhysioNet 2019: significance vs. proposed model (tcn_transformer) ===")
    print(physionet_sig.to_string(index=False))