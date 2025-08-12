#!/usr/bin/env python
"""Train a fully connected neural network and save per-sample explanations.

Explanations for each attribution method are written to
``results/explanations/FCNN/FCNN_*_{method}_L1_layer0_test.csv`` matching the
layout of BINN explanations so they can be summarized together.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from openbinn.explainer import Explainer
import openbinn.experiment_utils as utils
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

METHODS = ["itg", "ig", "gradshap", "deeplift", "shap"]
NO_BASELINE_METHODS = {"itg", "sg", "grad", "gradshap", "lrp", "lime", "control", "feature_ablation"}
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


def train_fnn(x_train, y_train, hidden_dim=128, epochs=50, batch_size=32, lr=1e-3):
    torch.manual_seed(SEED)
    input_dim = x_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb.reshape_as(logits))
            loss.backward()
            optimizer.step()
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

    model = train_fnn(x_tr, y_tr)
    model.eval()

    x_te = torch.from_numpy(x[ds.test_idx]).float()
    y_te = torch.from_numpy(y[ds.test_idx]).long()
    preds = model(x_te).detach().cpu().numpy().squeeze()

    n_genes = len(ds.node_index)
    n_feat = ds.x.shape[2]
    explain_root = data_dir / "results" / "explanations" / "FCNN"
    explain_root.mkdir(parents=True, exist_ok=True)
    config = utils.load_config(str(ROOT / "configs/experiment_config.json"))
    data_label = data_dir.name

    methods = [m for m in METHODS if m in config['explainers']]
    for method in methods:
        p_conf = utils.fill_param_dict(method, config['explainers'][method], x_te)
        p_conf['classification_type'] = 'binary'
        if method not in NO_BASELINE_METHODS:
            p_conf['baseline'] = torch.zeros_like(x_te)
        explainer = Explainer(method, model, p_conf)
        imp = explainer.get_explanations(x_te, y_te).detach().cpu().numpy()
        imp = imp.reshape(len(ds.test_idx), n_genes, n_feat).sum(axis=2)
        df = pd.DataFrame(imp, columns=list(ds.node_index))
        df.insert(0, 'sample_id', [str(i) for i in ds.test_idx])
        df['label'] = y_te.numpy()
        df['prediction'] = preds
        csv_fp = explain_root / f"FCNN_{data_label}_{method}_L1_layer0_test.csv"
        df.to_csv(csv_fp, index=False)
        print(f"Saved {method} explanations to {csv_fp}")


if __name__ == "__main__":
    main()
