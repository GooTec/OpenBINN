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

cnv_del_df = cnv_df.eq(-2).astype(int)
cnv_amp_df = cnv_df.eq(2).astype(int)

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

# ────────── (1b) Pathway helpers ──────────
def independent_paths(all_paths, thr: float = 0.2):
    """Return pathway IDs with mean Jaccard similarity < thr."""
    def jacc(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a | b) else 1.0

    keep = []
    for p, genes in all_paths.items():
        sims = [jacc(genes, all_paths[o]) for o in all_paths if o != p]
        if sims and np.mean(sims) < thr:
            keep.append(p)
    return keep


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
        mat = X[genes_subset].values
    else:
        mat = X.values
    pcs = PCA(n_components=2).fit_transform(mat)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["C0", "C1"]
    for cls in (0, 1):
        idx = y.values == cls
        ax[0].scatter(
            pcs[idx, 0],
            pcs[idx, 1],
            s=20,
            alpha=0.6,
            edgecolor="k",
            c=colors[cls],
            label=f"M={cls}",
        )
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("PCA")
    ax[0].legend(frameon=False)

    counts = y.value_counts().sort_index()
    ax[1].bar(counts.index.astype(str), counts.values, color=[colors[int(i)] for i in counts.index])
    ax[1].set_xlabel("y")
    ax[1].set_ylabel("count")
    ax[1].set_title("Outcome distribution")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

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

def main(start_sim: int = 1, end_sim: int = N_SIM, *,
         pathway_nonlinear: bool = False, alpha_sigma: float = 20.0,
         prev: float = 0.5):
    print("▶ Generating simulations & variants …")
    indep = independent_paths(pathways)
    for i in range(start_sim, end_sim + 1):
        rng_sim = np.random.RandomState(RNG_BASE_SEED + i)

        if pathway_nonlinear:
            # quadratic pathway model using provided beta/gamma
            S = X_true.values @ alpha
            p1 = BETA * S + GAMMA * (S ** 2)
            p2 = DELTAS[0] * p1
            p3 = DELTAS[1] * p2
            eta = p1 + p2 + p3
            c = calibrate_intercept(eta, prev)
            p = expit(eta + c)
            y = pd.Series(rng_sim.binomial(1, p), index=X_true.index, name="response")
            Xm_sel = mutation
            diag_df = pd.DataFrame({"id": X_true.index, "eta": eta, "prob": p, "response": y.values})
        else:
            true_p = rng_sim.choice(indep, 1)[0]
            nulls = list(rng_sim.choice([p for p in indep if p != true_p], 2, replace=False))
            pool = [true_p] + nulls

            genes = sorted({g for p in pool for g in pathways[p] if g in mutation.columns})
            Xm_sel = mutation[genes]

            a = {g: (rng_sim.normal(0, alpha_sigma) if g in pathways[true_p] else 0.0)
                 for g in genes}
            a_vec = np.array([a[g] for g in genes])
            additive = Xm_sel.values.dot(a_vec)

            tg = [g for g in genes if g in pathways[true_p]]
            if len(tg) >= 2:
                g1, g2 = rng_sim.choice(tg, 2, replace=False)
                mult = Xm_sel[g1].values * Xm_sel[g2].values
                OR = np.maximum(Xm_sel[g1].values, Xm_sel[g2].values)
                AND = np.minimum(Xm_sel[g1].values, Xm_sel[g2].values)
            else:
                mult = OR = AND = np.zeros(len(Xm_sel))

            w = rng_sim.uniform(size=4)
            eta = w[0] * additive + w[1] * mult + w[2] * OR + w[3] * AND
            c = calibrate_intercept(eta, prev)
            p = expit(eta + c)
            y = pd.Series(rng_sim.binomial(1, p), index=Xm_sel.index, name="response")
            diag_df = pd.DataFrame({"id": Xm_sel.index, "eta": eta, "prob": p, "response": y.values})

        sim_dir = OUT_ROOT / f"b{BETA}_g{GAMMA}" / f"{i}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        if pathway_nonlinear:
            genes_for_pca = list(true_genes)
        else:
            genes_for_pca = [g for g in pathways[true_p] if g in GA.columns]
        save_visualization(GA.loc[y.index], y, sim_dir / "pca_plot.png", genes_subset=genes_for_pca)
        save_triplet(sim_dir, mutation, cnv_aligned, y)
        (sim_dir/"selected_genes.csv").write_text(SELECTED_GENES_TXT)
        make_splits(y, sim_dir/"splits")
        diag_df.to_csv(sim_dir/"predictor_table.csv", index=False)
        with open(sim_dir/"intercept.txt", "w") as fh:
            fh.write(str(c))

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
    parser.add_argument("--start_sim", type=int, default=1,
                        help="Start index of simulation (inclusive)")
    parser.add_argument("--end_sim", type=int, default=N_SIM,
                        help="End index of simulation (inclusive)")
    parser.add_argument("--pathway_nonlinear", action="store_true",
                        help="Use pathway-based nonlinear outcome generation")
    parser.add_argument("--alpha_sigma", type=float, default=20.0,
                        help="Stddev of gene coefficients for true pathway")
    parser.add_argument("--prev", type=float, default=0.5,
                        help="Target prevalence when calibrating intercept")
    args = parser.parse_args()
    BETA = args.beta
    GAMMA = args.gamma
    main(args.start_sim, args.end_sim,
         pathway_nonlinear=args.pathway_nonlinear,
         alpha_sigma=args.alpha_sigma,
         prev=args.prev)
