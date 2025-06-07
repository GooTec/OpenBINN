#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_b4g4_simulations.py
───────────────────────────
β = 4, γ = 4 고정, 시뮬레이션 i=1‒100.
각 시뮬레이션 폴더 구조

data/b4_g4/{i}/
   ├─ somatic_mutation_paper.csv
   ├─ P1000_data_CNA_paper.csv
   ├─ response.csv
   ├─ selected_genes.csv
   ├─ splits/
   │    ├─ training_set_0.csv
   │    ├─ validation_set.csv
   │    └─ test_set.csv
   ├─ bootstrap/
   │    └─ {b=1‒100}/ (3파일+selected_genes.csv+splits)
   ├─ gene-permutation/
   │    └─ {b=1‒100}/ (〃)
   └─ label-permutation/
        └─ {b=1‒100}/ (〃)
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# ────────── 파라미터 ──────────
RNG_BASE_SEED = 42
N_SIM         = 100     # i = 1‒100
N_VARIANTS    = 100     # b = 1‒100
BETA, GAMMA   = 2, 2  # default values, can be overridden via CLI
DELTAS        = (0.5, 0.25)
OUT_ROOT      = Path("./data")

# ────────── (1) 공통 데이터 로드 ──────────
def read_gmt(fp: Path):
    d = {}
    with fp.open() as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                d[p[1]] = p[3:]
    return d

print("▶ Loading omics & pathway data …")
sim_kdir     = Path("../biological_knowledge/simulation")
pathways     = read_gmt(sim_kdir / "SimulationPathways.gmt")
TRUE_PWY     = ["R-HSA-2173791"]
true_genes   = {g for p in TRUE_PWY if p in pathways for g in pathways[p]}

mut_df = pd.read_csv("../data/prostate/P1000_final_analysis_set_cross_important_only.csv",
                     index_col=0)
cnv_df = pd.read_csv("../data/prostate/P1000_data_CNA_paper.csv", index_col=0)

cnv_del_df = cnv_df.applymap(lambda v: 1 if v == -2 else 0)
cnv_amp_df = cnv_df.applymap(lambda v: 1 if v ==  2 else 0)

ALL_GENES = sorted({g for genes in pathways.values() for g in genes})

def align_df(df, all_cols):
    """모든 유전자를 포함하도록 0-filled 열 추가."""
    return df.reindex(columns=all_cols, fill_value=0)

mutation = align_df(mut_df, ALL_GENES)
cnv_del  = align_df(cnv_del_df, ALL_GENES)
cnv_amp  = align_df(cnv_amp_df, ALL_GENES)
cnv_aligned = align_df(cnv_df, ALL_GENES)   # 저장용 원본 CNV 행렬

common_idx = mutation.index.intersection(cnv_aligned.index)
mutation, cnv_del, cnv_amp, cnv_aligned = (
    df.loc[common_idx] for df in (mutation, cnv_del, cnv_amp, cnv_aligned)
)

w = 1.5
GA = w*mutation + w*cnv_del + w*cnv_amp
X_true = GA.loc[:, GA.columns.intersection(true_genes)]
alpha  = np.ones(X_true.shape[1])

SELECTED_GENES_TXT = "genes\n" + "\n".join(mutation.columns)

# ────────── (2) 헬퍼 ──────────
def save_triplet(dst: Path, Xm, Xc, y):
    """
    • somatic_mutation_paper.csv  : index name → Tumor_Sample_Barcode
    • response.csv                : id, response 두 column
    """
    dst.mkdir(parents=True, exist_ok=True)

    # ─ somatic_mutation_paper.csv ─
    Xm2 = Xm.copy()
    Xm2.index.name = "Tumor_Sample_Barcode"
    Xm2.to_csv(dst / "somatic_mutation_paper.csv")

    # ─ P1000_data_CNA_paper.csv ─
    Xc.to_csv(dst / "P1000_data_CNA_paper.csv")

    # ─ response.csv ─  (id, response 두 컬럼)
    resp_df = y.to_frame(name="response").reset_index()          # index → column
    resp_df.rename(columns={"index": "id"}, inplace=True)
    resp_df.to_csv(dst / "response.csv", index=False)            # ⚠ index 저장 X


def make_splits(y, dst: Path, seed=42):
    dst.mkdir(parents=True, exist_ok=True)
    tr, tmp = train_test_split(y.index, test_size=0.2, stratify=y, random_state=seed)
    va, te  = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    def df(ids): return pd.DataFrame({'id': ids, 'response': y.loc[ids].values})
    (df(tr)).to_csv(dst/"training_set_0.csv", index=True)
    (df(va)).to_csv(dst/"validation_set.csv",  index=True)
    (df(te)).to_csv(dst/"test_set.csv",        index=True)

def make_bootstrap(Xm, Xc, y, rng):
    n = len(Xm); pos = rng.choice(n, size=n, replace=True)
    return (pd.DataFrame(Xm.values[pos], columns=Xm.columns, index=Xm.index),
            pd.DataFrame(Xc.values[pos], columns=Xc.columns, index=Xc.index),
            pd.Series(y.values[pos], index=y.index, name=y.name))

def make_gene_perm(Xm, Xc, y, rng):
    perm = rng.permutation(Xm.columns)
    Xm2, Xc2 = Xm.copy(), Xc.copy()
    Xm2.columns = Xc2.columns = perm
    return Xm2, Xc2, y

def make_label_perm(Xm, Xc, y, rng):
    yp = pd.Series(rng.permutation(y.values), index=y.index, name="response")
    return Xm, Xc, yp

def main():
    print("▶ Generating simulations & variants …")
    for i in range(1, N_SIM+1):
        rng_sim = np.random.RandomState(RNG_BASE_SEED + i)
        S   = X_true.values @ alpha
        eta = (BETA*S + GAMMA*S**2) + DELTAS[0]*(BETA*S + GAMMA*S**2) \
            +  DELTAS[1]*DELTAS[0]*(BETA*S + GAMMA*S**2)
        p   = 1/(1+np.exp(-eta))
        y   = pd.Series(rng_sim.binomial(1, p), index=X_true.index, name="response")

        sim_dir = OUT_ROOT / f"b{BETA}_g{GAMMA}" / f"{i}"
        save_triplet(sim_dir, mutation, cnv_aligned, y)
        (sim_dir/"selected_genes.csv").write_text(SELECTED_GENES_TXT)
        make_splits(y, sim_dir/"splits")

        for b in range(1, N_VARIANTS+1):
            bs_Xm, bs_Xc, bs_y = make_bootstrap(
                mutation, cnv_aligned, y,
                np.random.RandomState(RNG_BASE_SEED + i*10_000 + b)
            )
            bs_dir = sim_dir/"bootstrap"/f"{b}"
            save_triplet(bs_dir, bs_Xm, bs_Xc, bs_y)
            (bs_dir/"selected_genes.csv").write_text(SELECTED_GENES_TXT)
            make_splits(bs_y, bs_dir/"splits")

            gp_Xm, gp_Xc, gp_y = make_gene_perm(
                mutation, cnv_aligned, y,
                np.random.RandomState(RNG_BASE_SEED + i*20_000 + b)
            )
            gp_dir = sim_dir/"gene-permutation"/f"{b}"
            save_triplet(gp_dir, gp_Xm, gp_Xc, gp_y)
            gp_dir.joinpath("selected_genes.csv").write_text(
                "genes\n" + "\n".join(gp_Xm.columns)
            )
            make_splits(gp_y, gp_dir/"splits")

            lp_Xm, lp_Xc, lp_y = make_label_perm(
                mutation, cnv_aligned, y,
                np.random.RandomState(RNG_BASE_SEED + i*30_000 + b)
            )
            lp_dir = sim_dir/"label-permutation"/f"{b}"
            save_triplet(lp_dir, lp_Xm, lp_Xc, lp_y)
            (lp_dir/"selected_genes.csv").write_text(SELECTED_GENES_TXT)
            make_splits(lp_y, lp_dir/"splits")

        if i == 1 or i % 10 == 0:
            fpr, tpr, _ = roc_curve(y, p)
            auc_val = auc(fpr, tpr)
            print(f"  Sim {i:3d}| prev={y.mean():.3f}  AUC={auc_val:.3f}")

    print("✓ 모든 시뮬레이션·100개 변형·splits 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulation datasets")
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    args = parser.parse_args()
    BETA = args.beta
    GAMMA = args.gamma
    main()
