#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pvalue_calculation.py
---------------------
Compute null distributions and p-values for BINN importance scores.
The variant used to construct the null distribution is selected via
``--statistical_method``. Simulation data are read from a folder
``data/b{beta}_g{gamma}`` determined by the ``--beta`` and ``--gamma``
arguments.
"""

from pathlib import Path
import argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# user settings
METHOD = "deeplift"
N_VARIANTS = 100
DEFAULT_BETA  = 0
DEFAULT_GAMMA = 0.0
DATA_ROOT = Path(f"./data/b{DEFAULT_BETA}_g{DEFAULT_GAMMA}")
OUT_ROOT = Path(f"./results/b{DEFAULT_BETA}_g{DEFAULT_GAMMA}")
sns.set(style="whitegrid")

def process_sim(sim: int, variant: str):
    sim_dir = DATA_ROOT / f"{sim}"
    orig_fp = sim_dir / f"PNet_{METHOD}_target_scores.csv"
    if not orig_fp.exists():
        print(f"[skip] 원본 score 없음 → sim {sim}")
        return

    orig_df = pd.read_csv(orig_fp, index_col=0)

    perm_dfs = []
    for b in range(1, N_VARIANTS + 1):
        fp = sim_dir / variant / f"{b}" / f"PNet_{METHOD}_target_scores.csv"
        if fp.exists():
            perm_dfs.append(pd.read_csv(fp, index_col=0))
    if not perm_dfs:
        print(f"[skip] {variant} 없음 → sim {sim}")
        return

    for b, df_b in enumerate(perm_dfs, start=1):
        orig_df[f"null_{b}"] = df_b["importance"]

    pathways_df = orig_df[orig_df["layer"] > 0]
    df = pathways_df.reset_index().rename(columns={"index": "pathway_id"})
    df["uid"] = df["pathway_id"] + "_L" + df["layer"].astype(int).astype(str)
    df = df.set_index("uid")

    null_cols = [c for c in df.columns if c.startswith("null_")]
    null_list = [df.loc[r, null_cols].values.astype(float) for r in df.index]
    obs_list = df["importance"].values
    pathways = df.index.to_list()

    x_min = min([d.min() for d in null_list] + list(obs_list))
    x_max = max([d.max() for d in null_list] + list(obs_list))
    margin = 0.05 * (x_max - x_min)
    x_min -= margin
    x_max += margin

    fig, axes = plt.subplots(nrows=len(pathways), ncols=1,
                             figsize=(10, 4 * len(pathways)), sharex=True)
    for ax, pid, null_vals, obs in zip(axes, pathways, null_list, obs_list):
        sns.histplot(null_vals, bins=50, stat="density", kde=True, ax=ax)
        ax.axvline(obs, color="red", linestyle="--", label=f"Observed ({obs:.2f})")
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("Density")
        ax.set_title(pid)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Importance")
    plt.tight_layout()

    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_fp = out_dir / f"sim_{sim}_distributions.png"
    plt.savefig(fig_fp, dpi=300)
    plt.close(fig)

    p_vals = [np.mean(null >= obs) for null, obs in zip(null_list, obs_list)]
    p_df = pd.DataFrame({"pathway": pathways, "p_value": p_vals})
    csv_fp = out_dir / f"sim_{sim}_pvalues.csv"
    p_df.to_csv(csv_fp, index=False)

    print(f"{variant:<16} sim {sim:3d} │ saved → {fig_fp.name} , {csv_fp.name}")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--statistical_method",
                    choices=["bootstrap", "gene-permutation", "label-permutation"],
                    required=True,
                    help="Variant type used for the null distribution")
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim", type=int, default=100)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    args = ap.parse_args()

    global DATA_ROOT, OUT_ROOT
    DATA_ROOT = Path(f"./data/b{args.beta}_g{args.gamma}")
    OUT_ROOT = Path(f"./results/b{args.beta}_g{args.gamma}")

    for i in range(args.start_sim, args.end_sim + 1):
        process_sim(i, args.statistical_method)

    print("\n✓ p-value 계산 완료.")

if __name__ == "__main__":
    main()
