#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate bootstrap, gene-permutation, and label-permutation datasets
from a base simulation scenario, keeping the original row positions
and index labels fixed.

bootstrap: sample with replacement n rows, but always write out
           exactly n rows in original order, assigning sampled
           values into each original slot.
gene-perm: permute values within each gene column; row order untouched.
label-perm: permute response values; row order untouched.

Usage:
    python generate_variants.py --scenario /path/to/scenario
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_resample_fixed_positions(X: pd.DataFrame, y: pd.Series, rng: np.random.RandomState):
    n = len(X)
    # sample positions 0..n-1 with replacement
    pos = rng.choice(np.arange(n), size=n, replace=True)
    # build new DataFrame/Series with same index but values from sampled positions
    X_new = pd.DataFrame(X.values[pos, :], columns=X.columns, index=X.index)
    y_new = pd.Series(y.values[pos], index=y.index, name=y.name)
    return X_new, y_new


def gene_permutation(X: pd.DataFrame, rng: np.random.RandomState):
    Xp = X.copy()
    for col in Xp.columns:
        Xp[col] = rng.permutation(Xp[col].values)
    return Xp


def label_permutation(y: pd.Series, rng: np.random.RandomState):
    yp = y.copy()
    yp[:] = rng.permutation(yp.values)
    return yp


def main(scenario: Path):
    # expect these two files in scenario folder
    mut_fp = scenario / "somatic_mutation_paper.csv"
    resp_fp = scenario / "response.csv"
    if not mut_fp.exists() or not resp_fp.exists():
        raise FileNotFoundError(f"Need both {mut_fp.name} and {resp_fp.name} in {scenario}")

    # load original data, preserving order and index
    X = pd.read_csv(mut_fp, index_col=0)
    y = pd.read_csv(resp_fp).set_index("id")["response"]

    for method in ("bootstrap", "gene-perm", "label-perm"):
        out_base = scenario / method
        out_base.mkdir(parents=True, exist_ok=True)

        for seed in range(1001):
            rng = np.random.RandomState(seed)

            if method == "bootstrap":
                X_new, y_new = bootstrap_resample_fixed_positions(X, y, rng)
            elif method == "gene-perm":
                X_new = gene_permutation(X, rng)
                y_new = y
            else:  # label-perm
                X_new = X
                y_new = label_permutation(y, rng)

            subdir = out_base / str(seed)
            subdir.mkdir(exist_ok=True)

            # save with original index labels in same order
            X_new.to_csv(subdir / "somatic_mutation_paper.csv", index=True)
            y_new.to_frame().to_csv(subdir / "response.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bootstrap/gene-perm/label-perm variants preserving original row positions and index"
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Folder containing somatic_mutation_paper.csv and response.csv"
    )
    args = parser.parse_args()
    main(args.scenario)
