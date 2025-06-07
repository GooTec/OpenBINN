#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""generate_simulations.py
Generate simple simulation datasets and their bootstrap/gene-permutation/
label-permutation variants. The datasets are stored under
``data/b{beta}_g{gamma}/{sim}``.
This script is intentionally lightweight and does not replicate the full
simulation procedures used in the original project, but provides minimal
placeholders compatible with the training scripts.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def generate_single(out_dir: Path, beta: float, gamma: float, seed: int):
    rng = np.random.RandomState(seed)
    genes = [f"g{i+1}" for i in range(N_GENES)]

    X = rng.normal(0, 1, size=(N_SAMPLES, N_GENES))
    coefs = np.zeros(N_GENES)
    active = rng.choice(N_GENES, size=N_GENES // 2, replace=False)
    coefs[active] = rng.normal(beta, 0.1, size=len(active))
    eta = X.dot(coefs) + gamma
    p = 1 / (1 + np.exp(-eta))
    y = (rng.rand(N_SAMPLES) < p).astype(int)

    dfX = pd.DataFrame(X, columns=genes)
    dfX.index.name = "id"
    dfy = pd.Series(y, index=dfX.index, name="response")

    out_dir.mkdir(parents=True, exist_ok=True)
    dfX.to_csv(out_dir / "somatic_mutation_paper.csv")
    dfy.to_frame().to_csv(out_dir / "response.csv", index=True)
    pd.Series(genes, name="genes").to_csv(out_dir / "selected_genes.csv", index=False)
    pd.DataFrame({"gene": genes, "alpha": rng.normal(0, 1, size=N_GENES)}).to_csv(out_dir / "gene_alpha.csv", index=False)
    pd.DataFrame({"pathway": [f"p{i+1}" for i in range(len(active))], "beta": rng.normal(beta, 1, size=len(active))}).to_csv(out_dir / "pathway_beta.csv", index=False)

    tr, temp = train_test_split(dfX.index, train_size=0.8, random_state=seed)
    va, te = train_test_split(temp, train_size=0.5, random_state=seed)

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
    ap.add_argument("--n_sim", type=int, default=N_SIM)
    args = ap.parse_args()

    data_root = Path(f"./data/b{args.beta}_g{args.gamma}")
    for i in range(1, args.n_sim + 1):
        generate_single(data_root / f"{i}", args.beta, args.gamma, seed=42 + i)
    print(f"✓ generated simulations at {data_root}")


if __name__ == "__main__":
    main()
