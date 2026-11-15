"""
dp_sweep.py -- privacy-utility sweep: eps in {inf, 8, 3, 1} x seeds, FedAvg, same split/protocol.

    python dp_sweep.py --data windows.npz --model tcn_only --eps 8 3 1 --seeds 1000 1001 1002 1003 1004

eps = inf is the matched NON-PRIVATE baseline (same code path, same model, same everything).
Every row of runs/results_canonical.csv records: target eps, eps actually spent (Opacus RDP accountant,
max over clients), eps at the best round, sigma, delta, clip, rounds, local_epochs, state hash,
raw + calibrated Brier/ECE and per-site AUROC with class counts.

Interpretation notes for the paper:
  * the seeds are independent repetitions; each released model satisfies (eps, delta)-DP on its own
  * hyper-parameters were NOT tuned on private data here (fixed defaults) -- say so
  * record-level DP, not patient-level
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import fedavg as fa


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    g = defaultdict(list)
    for r in rows:
        if r["mode"] == "fedavg":
            g[(r["model"], r["eps_target"])].append(r)
    print("\n=== privacy-utility summary (mean +- sd over seeds) ===")
    print(f"{'model':16s}{'eps_target':>11s}{'eps_spent':>11s}{'AUROC':>16s}{'AUPRC':>16s}{'ECE_cal':>10s}{'n':>3s}")
    for (m, e), rs in sorted(g.items(), key=lambda kv: float(kv[0][1]), reverse=True):
        f = lambda k: np.array([float(r[k]) for r in rs])
        print(f"{m:16s}{e:>11s}{f('eps_spent_max').mean():11.3f}"
              f"{f('test_auroc').mean():9.4f}+-{f('test_auroc').std(ddof=1) if len(rs) > 1 else 0:.4f}"
              f"{f('test_auprc').mean():9.4f}+-{f('test_auprc').std(ddof=1) if len(rs) > 1 else 0:.4f}"
              f"{f('ece_cal').mean():10.4f}{len(rs):3d}")


def main():
    p = fa.add_args(argparse.ArgumentParser())
    p.add_argument("--eps", type=float, nargs="+", default=[8.0, 3.0, 1.0])
    p.add_argument("--no_baseline", action="store_true", help="skip eps=inf run")
    a = fa.apply_paper_config(p.parse_args())
    os.makedirs(a.out, exist_ok=True)
    a.log = os.path.join(a.out, "rounds.jsonl")
    data = fa.load_npz(a.data, a.seq_len, a.layout)
    a.n_features = a.n_features or data["train"]["X"].shape[2]
    budgets = ([] if a.no_baseline else [float("inf")]) + list(a.eps)
    for seed in a.seeds:
        for e in budgets:
            fa.run_one(data, a, "fedavg", seed, eps=e)
    summarize(os.path.join(a.out, "results_canonical.csv"))


if __name__ == "__main__":
    main()
