"""
Generates the same PNG + per-seed CSV chart output that the main
training script auto-saves for new models, but for models that
ALREADY have results sitting in all_model_seed_results_physionet2019.csv
(logistic_regression, xgboost, random_forest — trained before the
auto-chart feature existed).

Safe to run any time — it will (re)generate a chart for every model
found in the results CSV, using a fresh timestamp each time, so it
never overwrites or deletes anything.

Run:
    conda activate neuralicu-fix
    python generate_charts_from_csv.py
"""

import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_physionet2019 import DATA

RESULTS_DIR = f"{DATA['output_dir']}/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_seed_result_plot(model_name, seed_results_dict, metric_name="AUROC"):
    seeds = sorted(seed_results_dict.keys())
    values = [seed_results_dict[s] for s in seeds]
    mean_val = sum(values) / len(values)
    std_val = float(np.std(values))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seeds, values, marker="o", linewidth=1.5, color="#2C6E91")
    ax.axhline(mean_val, color="#C0392B", linestyle="--",
               label=f"mean = {mean_val:.4f}  (SD = {std_val:.4f})")
    ax.set_xlabel("Seed")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{model_name} — {metric_name} across {len(seeds)} seeds")
    ax.legend()
    ax.grid(alpha=0.3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(RESULTS_DIR, f"{model_name}_{metric_name}_{timestamp}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(RESULTS_DIR, f"{model_name}_{metric_name}_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", metric_name])
        for s, v in zip(seeds, values):
            writer.writerow([s, v])

    print(f"  Chart saved -> {png_path}")
    print(f"  Per-seed CSV saved -> {csv_path}")
    return png_path


def main():
    results_path = f"{DATA['output_dir']}/all_model_seed_results_physionet2019.csv"
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}")
        return

    df = pd.read_csv(results_path)
    models_found = sorted(df["model"].unique())
    print(f"Found {len(models_found)} models in results file: {models_found}\n")

    for model_name in models_found:
        model_df = df[df["model"] == model_name]
        seed_results = dict(zip(model_df["seed"], model_df["auroc"]))
        print(f"=== {model_name} — {len(seed_results)} seeds ===")
        save_seed_result_plot(model_name, seed_results)
        print()

    # Also build one summary bar chart comparing all models' mean AUROC
    summary = df.groupby("model")["auroc"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["model"], summary["mean"], yerr=summary["std"],
           color="#2C6E91", capsize=5)
    ax.set_ylabel("AUROC (mean ± SD)")
    ax.set_title("Model comparison — mean AUROC across seeds")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_png = os.path.join(RESULTS_DIR, f"model_comparison_summary_{timestamp}.png")
    fig.savefig(summary_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary comparison chart saved -> {summary_png}")

    summary_csv = os.path.join(RESULTS_DIR, f"model_comparison_summary_{timestamp}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Summary comparison CSV saved -> {summary_csv}")


if __name__ == "__main__":
    main()
