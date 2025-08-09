#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from scipy.special import expit
import matplotlib.pyplot as plt
import json

# ────────── 파라미터 ──────────
RNG_BASE_SEED = 42
N_SIM         = 100
PATHWAY_LINEAR_EFFECT, PATHWAY_NONLINEAR_EFFECT = 2, 2
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
TRUE_PWY_ID  = "R-HSA-2173791"
true_genes   = set(pathways.get(TRUE_PWY_ID, []))

mut_df = pd.read_csv("../data/prostate/P1000_final_analysis_set_cross_important_only.csv", index_col=0)
cnv_df = pd.read_csv("../data/prostate/P1000_data_CNA_paper.csv", index_col=0)

cnv_del_df = cnv_df.eq(-2).astype(int)
cnv_amp_df = cnv_df.eq(2).astype(int)

ALL_GENES = sorted({g for genes in pathways.values() for g in genes})

def align_df(df, all_cols):
    return df.reindex(columns=all_cols, fill_value=0)

mutation = align_df(mut_df, ALL_GENES)
cnv_del  = align_df(cnv_del_df, ALL_GENES)
cnv_amp  = align_df(cnv_amp_df, ALL_GENES)
cnv_aligned = align_df(cnv_df, ALL_GENES)

common_idx = mutation.index.intersection(cnv_aligned.index)
mutation, cnv_del, cnv_amp, cnv_aligned = (df.loc[common_idx] for df in (mutation, cnv_del, cnv_amp, cnv_aligned))

omics_effect = {"mutation": 1.0, "cnv_del": 1.0, "cnv_amp": 1.0}
GA = (
    omics_effect["mutation"] * mutation
    + omics_effect["cnv_del"] * cnv_del
    + omics_effect["cnv_amp"] * cnv_amp
)

X_true = GA.loc[:, GA.columns.intersection(true_genes)]

# ────────── (1b) Pathway helpers ──────────
def calibrate_intercept(eta: np.ndarray, prev: float) -> float:
    lo, hi = -20, 20
    for _ in range(40):
        mid = (lo + hi) / 2
        if expit(eta + mid).mean() > prev:
            hi = mid
        else:
            lo = mid
    return mid

# ────────── Visualization ──────────
def save_visualization(X: pd.DataFrame, y: pd.Series, out: Path, *, genes_subset=None):
    if genes_subset is not None:
        mat = X[[g for g in genes_subset if g in X.columns]].values
    else:
        mat = X.values
    pcs = PCA(n_components=2).fit_transform(mat)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["C0", "C1"]
    for cls in (0, 1):
        idx = y.values == cls
        ax[0].scatter(pcs[idx, 0], pcs[idx, 1], s=20, alpha=0.6, edgecolor="k", c=colors[cls], label=f"M={cls}")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].set_title("PCA")
    ax[0].legend(frameon=False)
    counts = y.value_counts().sort_index()
    ax[1].bar(counts.index.astype(str), counts.values, color=[colors[int(i)] for i in counts.index])
    ax[1].set_xlabel("y"); ax[1].set_ylabel("count"); ax[1].set_title("Outcome distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

# ────────── (2) 헬퍼 ──────────
def save_triplet(dst: Path, Xm, Xc, y):
    dst.mkdir(parents=True, exist_ok=True)
    Xm2 = Xm.copy(); Xm2.index.name = "Tumor_Sample_Barcode"
    Xm2.to_csv(dst / "somatic_mutation_paper.csv")
    Xc.to_csv(dst / "P1000_data_CNA_paper.csv")
    resp_df = y.to_frame(name="response").reset_index().rename(columns={"index": "id"})
    resp_df.to_csv(dst / "response.csv", index=False)

def make_splits(y, dst: Path, seed=42):
    dst.mkdir(parents=True, exist_ok=True)
    tr, tmp = train_test_split(y.index, test_size=0.2, stratify=y, random_state=seed)
    va, te  = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    def df(ids): return pd.DataFrame({'id': ids, 'response': y.loc[ids].values})
    (df(tr)).to_csv(dst/"training_set_0.csv", index=True)
    (df(va)).to_csv(dst/"validation_set.csv",  index=True)
    (df(te)).to_csv(dst/"test_set.csv",        index=True)

def main(start_sim: int = 1, end_sim: int = N_SIM, *,
         pathway_nonlinear: bool = False, gene_effect_sigma: float = 0.0,
         prev: float = 0.5):
    print("▶ Generating noise-free simulations …")
    for i in range(start_sim, end_sim + 1):
        rng_sim = np.random.RandomState(RNG_BASE_SEED + i)

        gene_effect = np.ones(X_true.shape[1])
        pathway_score = X_true.values @ gene_effect
        
        eta = PATHWAY_LINEAR_EFFECT * pathway_score
        if pathway_nonlinear:
            eta += PATHWAY_NONLINEAR_EFFECT * (pathway_score ** 2)

        c = calibrate_intercept(eta, prev)
        p = expit(eta + c)
        y = pd.Series(rng_sim.binomial(1, p), index=X_true.index, name="response")
        
        diag_df = pd.DataFrame({"id": X_true.index, "eta": eta, "prob": p, "response": y.values})
        sim_dir = OUT_ROOT / f"b{PATHWAY_LINEAR_EFFECT}_g{PATHWAY_NONLINEAR_EFFECT}" / f"{i}"

        sim_dir.mkdir(parents=True, exist_ok=True)
        save_visualization(GA.loc[y.index], y, sim_dir / "pca_plot.png", genes_subset=list(true_genes))
        save_triplet(sim_dir, mutation, cnv_aligned, y)
        make_splits(y, sim_dir/"splits")
        diag_df.to_csv(sim_dir/"predictor_table.csv", index=False)
        
        ground_truth = {
            "true_pathway_id": TRUE_PWY_ID,
            "true_genes": sorted(list(true_genes)),
            "pathway_linear_effect": PATHWAY_LINEAR_EFFECT,
            "pathway_nonlinear_effect": PATHWAY_NONLINEAR_EFFECT if pathway_nonlinear else 0
        }
        with open(sim_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=4)
        
        # <<< 에러 해결을 위해 추가된 부분 >>>
        # 데이터 로더가 요구하는 '전체 유전자 목록' 파일을 생성합니다.
        selected_genes_text = "genes\n" + "\n".join(ALL_GENES)
        (sim_dir / "selected_genes.csv").write_text(selected_genes_text)
        
        if i == 1 or i % 10 == 0:
            fpr, tpr, _ = roc_curve(y, p)
            auc_val = auc(fpr, tpr)
            print(f"  Sim {i:3d}| prev={y.mean():.3f}  AUC={auc_val:.3f}")

    print("✓ 모든 시뮬레이션 데이터 및 splits 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulation datasets")
    parser.add_argument("--pathway_linear_effect", "--beta", dest="pathway_linear_effect", type=float, default=PATHWAY_LINEAR_EFFECT)
    parser.add_argument("--pathway_nonlinear_effect", "--gamma", dest="pathway_nonlinear_effect", type=float, default=PATHWAY_NONLINEAR_EFFECT)
    parser.add_argument("--start_sim", type=int, default=1)
    parser.add_argument("--end_sim", type=int, default=N_SIM)
    parser.add_argument("--pathway_nonlinear", action="store_true", help="Use pathway-based nonlinear outcome generation")
    parser.add_argument("--gene_effect_sigma", type=float, default=0.0, help="This argument is no longer used in the simplified script.") 
    parser.add_argument("--prev", type=float, default=0.5, help="Target prevalence when calibrating intercept")
    
    args = parser.parse_args()
    PATHWAY_LINEAR_EFFECT = args.pathway_linear_effect
    PATHWAY_NONLINEAR_EFFECT = args.pathway_nonlinear_effect
    main(args.start_sim, args.end_sim,
         pathway_nonlinear=args.pathway_nonlinear,
         gene_effect_sigma=args.gene_effect_sigma,
         prev=args.prev)