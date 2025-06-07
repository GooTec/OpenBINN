#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_geneperm_pvalues.py
───────────────────────────
• 원본 vs gene-permutation 100개로 null-distribution 작성
• p-values (P[ null ≥ observed ]) 계산
• 그림 + CSV 저장
      results/gene_permutation/sim_{i}_distributions.png
      results/gene_permutation/sim_{i}_pvalues.csv
"""

from pathlib import Path
import argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ───────── 사용자 설정 ─────────
METHOD        = "deeplift"
N_VARIANTS    = 100
DATA_ROOT     = Path("./data/b0_g0.0")
OUT_DIR       = Path("./results/b0_g0.0/gene_permutation")
sns.set(style="whitegrid")
# ──────────────────────────────


def process_sim(i: int):
    sim_dir = DATA_ROOT / f"{i}"
    orig_fp = sim_dir / f"PNet_{METHOD}_target_scores.csv"
    if not orig_fp.exists():
        print(f"[skip] 원본 score 없음 → sim {i}")
        return

    # → 원본
    orig_df = pd.read_csv(orig_fp, index_col=0)

    # → gene-perm 100개
    boot_dfs = []
    for b in range(1, N_VARIANTS + 1):
        fp = sim_dir / "gene-permutation" / f"{b}" / f"PNet_{METHOD}_target_scores.csv"
        if fp.exists():
            boot_dfs.append(pd.read_csv(fp, index_col=0))
    if not boot_dfs:
        print(f"[skip] gene-perm 없음 → sim {i}")
        return

    # 열 결합
    for b, df_b in enumerate(boot_dfs, start=1):
        orig_df[f'bootstrap_{b}'] = df_b['importance']
    pathways_df = orig_df[orig_df['layer'] > 0]

    df = (pathways_df.reset_index()
          .rename(columns={'index': 'pathway_id'}))
    df['uid'] = df['pathway_id'] + '_L' + df['layer'].astype(int).astype(str)
    df = df.set_index('uid')

    boot_cols = [c for c in df.columns if c.startswith('bootstrap_')]
    data_list = [df.loc[r, boot_cols].values.astype(float) for r in df.index]
    obs_list  = df['importance'].values
    pathways  = df.index.to_list()

    # 그림
    x_min = min([d.min() for d in data_list] + list(obs_list))
    x_max = max([d.max() for d in data_list] + list(obs_list))
    margin = 0.05 * (x_max - x_min); x_min -= margin; x_max += margin

    fig, axes = plt.subplots(
        nrows=len(pathways), ncols=1,
        figsize=(10, 4 * len(pathways)), sharex=True
    )
    for ax, p_id, null_vals, obs in zip(axes, pathways, data_list, obs_list):
        sns.histplot(null_vals, bins=50, stat='density', kde=True, ax=ax)
        ax.axvline(obs, color='red', linestyle='--',
                   label=f'Observed ({obs:.2f})')
        ax.set_xlim(x_min, x_max); ax.set_ylabel('Density')
        ax.set_title(p_id); ax.legend(frameon=False)
    axes[-1].set_xlabel('Importance')
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_fp = OUT_DIR / f"sim_{i}_distributions.png"
    plt.savefig(fig_fp, dpi=300); plt.close(fig)

    # p-values
    p_vals = [np.mean(null >= obs) for null, obs in zip(data_list, obs_list)]
    p_df = pd.DataFrame({'pathway': pathways, 'p_value': p_vals})
    csv_fp = OUT_DIR / f"sim_{i}_pvalues.csv"
    p_df.to_csv(csv_fp, index=False)

    print(f"sim {i:3d} │ saved  → {fig_fp.name} , {csv_fp.name}")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim",   type=int, default=100)
    args = ap.parse_args()

    for i in range(args.start_sim, args.end_sim + 1):
        process_sim(i)

    print("\n✓ gene-permutation null-distribution & p-value 계산 완료.")


if __name__ == "__main__":
    main()
