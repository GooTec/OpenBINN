#!/usr/bin/env python
"""Train a logistic regression on simulation data and save per-gene betas.

The coefficients are written to ``results/explanations/Logistic/logistic_beta.csv``
so that ``feature_importance_summary.py`` can aggregate them alongside FCNN and
PNET explanations.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

# ensure repository root is on sys.path so ``openbinn`` can be imported when
# executing this script from within the ``analysis`` directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

SEED = 42


def load_dataset(data_dir: Path):
    ds = PnetSimDataSet(root=str(data_dir), num_features=3)
    ds.split_index_by_file(
        train_fp=data_dir / "splits" / "training_set_0.csv",
        valid_fp=data_dir / "splits" / "validation_set.csv",
        test_fp=data_dir / "splits" / "test_set.csv",
    )
    reactome = ReactomeNetwork(
        dict(
            reactome_base_dir="../biological_knowledge/simulation",
            relations_file_name="SimulationPathwaysRelation.txt",
            pathway_names_file_name="SimulationPathways.txt",
            pathway_genes_file_name="SimulationPathways.gmt",
        )
    )
    maps = get_layer_maps(
        genes=list(ds.node_index),
        reactome=reactome,
        n_levels=3,
        direction="root_to_leaf",
        add_unk_genes=False,
    )
    ds.align_with_map(maps[0].index)
    x = ds.x.reshape(ds.x.shape[0], -1).numpy()
    y = ds.y.reshape(-1).numpy()
    return ds, x, y


def train_logistic(x_train, y_train):
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=SEED)
    model.fit(x_train, y_train)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--rep", type=int, default=1)
    args = ap.parse_args()

    data_dir = Path(f"./data/experiment1/b{args.beta}_g{args.gamma}/{args.rep}")
    ds, x, y = load_dataset(data_dir)
    x_tr, y_tr = x[ds.train_idx], y[ds.train_idx]

    model = train_logistic(x_tr, y_tr)

    # save trained model for later evaluation/comparison
    opt_dir = data_dir / "results" / "optimal"
    opt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, opt_dir / "logistic_model.joblib")

    n_genes = len(ds.node_index)
    n_feat = ds.x.shape[2]
    beta = model.coef_[0].reshape(n_genes, n_feat).sum(axis=1)

    exp_dir = data_dir / "results" / "explanations" / "Logistic"
    exp_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"gene": ds.node_index, "importance": beta})
    df.to_csv(exp_dir / "logistic_beta.csv", index=False)
    print("Saved logistic betas to", exp_dir / "logistic_beta.csv")


if __name__ == "__main__":
    main()
