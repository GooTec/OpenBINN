import sys
from pathlib import Path

# Ensure local package resolution when running from repository root
cwd = Path.cwd()
if (cwd / "openbinn").exists():
    sys.path.insert(0, str(cwd))
elif (cwd.parent / "openbinn").exists():
    sys.path.insert(0, str(cwd.parent))

import argparse
import csv
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import joblib

from openbinn.binn import PNet
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

from experiment1_constants import (
    RESULTS_DIR,
    OPTIMAL_DIR,
    LOGISTIC_MODEL_FILENAME,
    FCNN_MODEL_FILENAME,
    PNET_MODEL_FILENAME,
    PNET_CONFIG_FILENAME,
    FCNN_HIDDEN_DIM,
    DEFAULT_BATCH_SIZE,
)


def load_dataset(data_dir: Path):
    """Load simulation dataset and align feature order with pathway maps."""
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
    x = ds.x.reshape(len(ds.y), -1).numpy()
    y = ds.y.numpy().ravel()
    return ds, x, y, maps


def load_logistic(model_dir: Path):
    fp = model_dir / LOGISTIC_MODEL_FILENAME
    if not fp.exists():
        raise FileNotFoundError(fp)
    return joblib.load(fp)


def load_fnn(model_dir: Path, input_dim: int):
    fp = model_dir / FCNN_MODEL_FILENAME
    if not fp.exists():
        raise FileNotFoundError(fp)
    ckpt = torch.load(fp, map_location="cpu")
    hidden_dim = ckpt.get("hidden_dim", FCNN_HIDDEN_DIM)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_pnet(model_dir: Path, maps):
    ckpt_fp = model_dir / PNET_MODEL_FILENAME
    cfg_fp = model_dir / PNET_CONFIG_FILENAME
    if not ckpt_fp.exists():
        raise FileNotFoundError(f"Missing PNet trained model checkpoint: {ckpt_fp}")
    if not cfg_fp.exists():
        raise FileNotFoundError(f"Missing PNet config file: {cfg_fp}")
    with open(cfg_fp) as f:
        cfg = json.load(f)
    optim_cfg = {
        "opt": cfg.get("optimizer", "adam"),
        "lr": cfg.get("learning_rate", 1e-3),
        "wd": cfg.get("weight_decay", 0.0),
        "scheduler": cfg.get("scheduler", "none"),
        "monitor": "val_auc",
    }
    model = PNet(
        layers=maps,
        num_genes=maps[0].shape[0],
        lr=cfg.get("learning_rate", 1e-3),
        norm_type=cfg.get("norm_type", "batchnorm"),
        dropout_rate=cfg.get("dropout_rate", 0.1),
        input_dropout=cfg.get("input_dropout", 0.0),
        loss_cfg=cfg.get("loss_cfg", {"main": 1.0, "aux": 0.0}),
        optim_cfg=optim_cfg,
    )
    state = torch.load(ckpt_fp, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def evaluate_auc_pnet(model, ds, indices, batch_size=DEFAULT_BATCH_SIZE):
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size)
    all_probs, all_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)[-1].squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_true.append(yb.view(-1).cpu())
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_true).numpy()
    return roc_auc_score(y_true, probs)


def evaluate_auc(model, x, y, is_torch=False):
    if is_torch:
        with torch.no_grad():
            logits = model(torch.from_numpy(x).float()).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
    else:
        probs = model.predict_proba(x)[:, 1]
    return roc_auc_score(y, probs)


def save_params(model, out_dir: Path, name: str, is_torch=False):
    if is_torch:
        torch.save(model.state_dict(), out_dir / f"{name}_params.pt")
    else:
        np.save(out_dir / f"{name}_coef.npy", model.coef_)
        np.save(out_dir / f"{name}_intercept.npy", model.intercept_)


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, x, y, maps = load_dataset(Path(args.data_dir))
    tr_idx, va_idx, te_idx = ds.train_idx, ds.valid_idx, ds.test_idx
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[va_idx], y[va_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    model_dir = Path(args.data_dir) / RESULTS_DIR / OPTIMAL_DIR

    # Logistic regression
    log_model = load_logistic(model_dir)
    log_res = {
        "train_auc": evaluate_auc(log_model, x_tr, y_tr),
        "val_auc": evaluate_auc(log_model, x_va, y_va),
        "test_auc": evaluate_auc(log_model, x_te, y_te),
    }
    save_params(log_model, out_dir, "logreg")
    with open(out_dir / "logreg_metrics.json", "w") as f:
        json.dump(log_res, f, indent=2)

    # Fully connected NN
    fnn_model = load_fnn(model_dir, x.shape[1])
    fnn_res = {
        "train_auc": evaluate_auc(fnn_model, x_tr, y_tr, is_torch=True),
        "val_auc": evaluate_auc(fnn_model, x_va, y_va, is_torch=True),
        "test_auc": evaluate_auc(fnn_model, x_te, y_te, is_torch=True),
    }
    save_params(fnn_model, out_dir, "fnn", is_torch=True)
    with open(out_dir / "fnn_metrics.json", "w") as f:
        json.dump(fnn_res, f, indent=2)

    # PNet model
    pnet_model, pnet_cfg = load_pnet(model_dir, maps)
    batch_size = pnet_cfg.get("batch_size", DEFAULT_BATCH_SIZE)
    pnet_res = {
        "train_auc": evaluate_auc_pnet(pnet_model, ds, tr_idx, batch_size=batch_size),
        "val_auc": evaluate_auc_pnet(pnet_model, ds, va_idx, batch_size=batch_size),
        "test_auc": evaluate_auc_pnet(pnet_model, ds, te_idx, batch_size=batch_size),
    }
    save_params(pnet_model, out_dir, "pnet", is_torch=True)
    with open(out_dir / "pnet_metrics.json", "w") as f:
        json.dump(pnet_res, f, indent=2)

    # Aggregate results
    rows = [
        {"model": "logistic_regression", **log_res},
        {"model": "fcnn", **fnn_res},
        {"model": "pnet", **pnet_res},
    ]
    with open(out_dir / "model_auc.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "train_auc", "val_auc", "test_auc"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare logistic regression, FCNN, and PNet on simulation data")
    parser.add_argument("--data-dir", default="./data/prostate", help="Directory containing simulation dataset")
    parser.add_argument("--output-dir", default="experiment/comparison", help="Where to store results")
    args = parser.parse_args()
    main(args)
