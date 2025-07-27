#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""generate_simulations.py
Generate simple simulation datasets and their bootstrap/gene-permutation/
label-permutation variants. The datasets are stored under
``data/experiment{exp}/b{beta}_g{gamma}/{sim}`` if ``--exp`` is provided,
otherwise ``data/b{beta}_g{gamma}/{sim}``.
This script is intentionally lightweight and does not replicate the full
simulation procedures used in the original project, but provides minimal
placeholders compatible with the training scripts.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.special import expit
import matplotlib.pyplot as plt

N_SIM = 100
N_VARIANTS = 100
N_SAMPLES = 200
N_GENES = 20


def bootstrap_fixed(X: pd.DataFrame, y: pd.Series, rng: np.random.RandomState):
    idx = rng.choice(len(X), size=len(X), replace=True)
    Xb = pd.DataFrame(X.values[idx, :], columns=X.columns, index=X.index)
    yb = pd.Series(y.values[idx], index=y.index, name=y.name)
    return Xb, yb


def gene_permutation(X: pd.DataFrame, rng: np.random.RandomState):
    Xp = X.copy()
    for col in Xp.columns:
        Xp[col] = rng.permutation(Xp[col].values)
    return Xp


def label_permutation(y: pd.Series, rng: np.random.RandomState):
    yp = y.copy()
    yp[:] = rng.permutation(yp.values)
    return yp


def calibrate_intercept(eta: np.ndarray, prev: float) -> float:
    """Binary intercept calibration for desired prevalence."""
    lo, hi = -20, 20
    for _ in range(40):
        mid = (lo + hi) / 2
        if expit(eta + mid).mean() > prev:
            hi = mid
        else:
            lo = mid
    return mid


def save_visualization(
    X: pd.DataFrame,
    y: pd.Series,
    out_path: Path,
    *,
    genes_subset: Optional[List[str]] = None,
) -> None:
    """Save PCA scatter with outcome distribution.

    If ``genes_subset`` is provided, PCA is computed using only those genes.
    """
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_single(
    out_dir: Path,
    beta: float,
    gamma: float,
    seed: int,
    *,
    pathway_nonlinear: bool = False,
    alpha_sigma: float = 1.0,
    prev: float = 0.5,
):
    rng = np.random.RandomState(seed)
    genes = [f"g{i+1}" for i in range(N_GENES)]

    X = rng.normal(0, 1, size=(N_SAMPLES, N_GENES))
    dfX = pd.DataFrame(X, columns=genes)
    dfX.index.name = "id"
    true_genes: list[str] = []

    if pathway_nonlinear:
        true_sz = min(5, N_GENES)
        true_genes = rng.choice(genes, size=true_sz, replace=False)
        remaining = [g for g in genes if g not in true_genes]
        null1 = rng.choice(remaining, size=min(true_sz, len(remaining)), replace=False)
        remaining = [g for g in remaining if g not in null1]
        null2 = rng.choice(remaining, size=min(true_sz, len(remaining)), replace=False)

        # apply the supplied beta value directly for gene effects
        alpha = {g: (beta if g in true_genes else 0.0) for g in genes}
        a_vec = np.array([alpha[g] for g in genes])
        additive = dfX.values.dot(a_vec)

        if len(true_genes) >= 2:
            g1, g2 = rng.choice(true_genes, 2, replace=False)
            mult = dfX[g1].values * dfX[g2].values
            OR = np.maximum(dfX[g1].values, dfX[g2].values)
            AND = np.minimum(dfX[g1].values, dfX[g2].values)
        else:
            mult = OR = AND = np.zeros(len(dfX))

        # use supplied beta/gamma for additive and nonlinear components
        eta = (
            beta * additive
            + gamma * mult
            + gamma * OR
            + gamma * AND
        )
        c = calibrate_intercept(eta, prev)
        prob = expit(eta + c)
        y = rng.binomial(1, prob)
        dfy = pd.Series(y, index=dfX.index, name="response")
        pathway_beta = pd.DataFrame(
            {
                "pathway": ["p_true", "p_null1", "p_null2"],
                "beta": [gamma, 0.0, 0.0],
            }
        )
    else:
        coefs = np.zeros(N_GENES)
        active = rng.choice(N_GENES, size=N_GENES // 2, replace=False)
        coefs[active] = beta
        eta = dfX.values.dot(coefs)
        prob = expit(eta)
        y = (rng.rand(N_SAMPLES) < prob).astype(int)
        dfy = pd.Series(y, index=dfX.index, name="response")
        alpha = {g: coefs[i] for i, g in enumerate(genes)}
        pathway_beta = pd.DataFrame(
            {
                "pathway": [f"p{i+1}" for i in range(len(active))],
                "beta": [gamma] * len(active),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    subset = true_genes if pathway_nonlinear else None
    save_visualization(dfX, dfy, out_dir / "pca_plot.png", genes_subset=subset)
    dfX.to_csv(out_dir / "somatic_mutation_paper.csv")
    dfy.to_frame().to_csv(out_dir / "response.csv", index=True)
    pd.Series(genes, name="genes").to_csv(out_dir / "selected_genes.csv", index=False)
    pd.DataFrame({"gene": genes, "alpha": [alpha.get(g, 0.0) for g in genes]}).to_csv(
        out_dir / "gene_alpha.csv", index=False
    )
    pathway_beta.to_csv(out_dir / "pathway_beta.csv", index=False)

    # stratified splits to maintain consistent outcome distribution across sets
    tr, temp = train_test_split(
        dfX.index,
        train_size=0.8,
        random_state=seed,
        stratify=dfy,
    )
    va, te = train_test_split(
        temp,
        train_size=0.5,
        random_state=seed,
        stratify=dfy.loc[temp],
    )

    sp = out_dir / "splits"
    sp.mkdir(exist_ok=True)
    pd.DataFrame({"id": tr, "response": dfy.loc[tr]}).to_csv(sp / "training_set_0.csv", index=True)
    pd.DataFrame({"id": va, "response": dfy.loc[va]}).to_csv(sp / "validation_set.csv", index=True)
    pd.DataFrame({"id": te, "response": dfy.loc[te]}).to_csv(sp / "test_set.csv", index=True)

    for method in ("bootstrap", "gene-permutation", "label-permutation"):
        base = out_dir / method
        base.mkdir(exist_ok=True)
        for b in range(1, N_VARIANTS + 1):
            sub = base / f"{b}"
            sub.mkdir(parents=True, exist_ok=True)
            rng_v = np.random.RandomState(seed + b)
            if method == "bootstrap":
                Xv, yv = bootstrap_fixed(dfX, dfy, rng_v)
            elif method == "gene-permutation":
                Xv = gene_permutation(dfX, rng_v)
                yv = dfy
            else:
                Xv = dfX
                yv = label_permutation(dfy, rng_v)
            Xv.to_csv(sub / "somatic_mutation_paper.csv")
            yv.to_frame().to_csv(sub / "response.csv", index=True)
            spv = sub / "splits"
            spv.mkdir(exist_ok=True)
            pd.DataFrame({"id": tr, "response": yv.loc[tr]}).to_csv(spv / "training_set_0.csv", index=True)
            pd.DataFrame({"id": va, "response": yv.loc[va]}).to_csv(spv / "validation_set.csv", index=True)
            pd.DataFrame({"id": te, "response": yv.loc[te]}).to_csv(spv / "test_set.csv", index=True)


def main():
    ap = argparse.ArgumentParser(description="Generate simulation datasets")
    ap.add_argument("--beta", type=float, default=0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--n_sim", type=int, default=N_SIM,
                    help="Number of simulations if --end_sim is not given")
    ap.add_argument("--start_sim", type=int, default=1,
                    help="Start index of simulation (inclusive)")
    ap.add_argument("--end_sim", type=int,
                    help="End index of simulation (inclusive). Defaults to n_sim")
    ap.add_argument("--exp", type=int, default=None,
                    help="Experiment number to store data under")
    ap.add_argument("--pathway_nonlinear", action="store_true",
                    help="Use pathway-driven nonlinear outcome")
    ap.add_argument("--alpha_sigma", type=float, default=1.0,
                    help="Stddev for gene coefficients of the true pathway")
    ap.add_argument("--prev", type=float, default=0.5,
                    help="Desired prevalence when calibrating intercept")
    args = ap.parse_args()

    end = args.end_sim if args.end_sim is not None else args.n_sim
    if end < args.start_sim:
        raise ValueError("end_sim must be >= start_sim")

    data_root = Path("./data")
    if args.exp is not None:
        data_root = data_root / f"experiment{args.exp}"
    data_root = data_root / f"b{args.beta}_g{args.gamma}"
    for i in range(args.start_sim, end + 1):
        generate_single(
            data_root / f"{i}",
            args.beta,
            args.gamma,
            seed=42 + i,
            pathway_nonlinear=args.pathway_nonlinear,
            alpha_sigma=args.alpha_sigma,
            prev=args.prev,
        )
    print(f"✓ generated simulations at {data_root}")


if __name__ == "__main__":
    main()
