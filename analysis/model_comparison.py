import sys
from pathlib import Path

# Ensure local package resolution when running from repository root
cwd = Path.cwd()
if (cwd / "openbinn").exists():
    sys.path.insert(0, str(cwd))
elif (cwd.parent / "openbinn").exists():
    sys.path.insert(0, str(cwd.parent))

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps


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
    return ds, x, y


def train_logistic(x_train, y_train):
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    model.fit(x_train, y_train)
    return model


def train_fnn(x_train, y_train, hidden_dim=128, epochs=50, batch_size=32, lr=1e-3):
    input_dim = x_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb.reshape_as(logits))
            loss.backward()
            optimizer.step()
    return model


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

    ds, x, y = load_dataset(Path(args.data_dir))
    tr_idx, va_idx, te_idx = ds.train_idx, ds.valid_idx, ds.test_idx
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[va_idx], y[va_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    # Logistic regression
    log_model = train_logistic(x_tr, y_tr)
    log_res = {
        "train_auc": evaluate_auc(log_model, x_tr, y_tr),
        "val_auc": evaluate_auc(log_model, x_va, y_va),
        "test_auc": evaluate_auc(log_model, x_te, y_te),
    }
    save_params(log_model, out_dir, "logreg")
    with open(out_dir / "logreg_metrics.json", "w") as f:
        json.dump(log_res, f, indent=2)

    # Fully connected NN
    fnn_model = train_fnn(x_tr, y_tr, hidden_dim=args.hidden_dim, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    fnn_res = {
        "train_auc": evaluate_auc(fnn_model, x_tr, y_tr, is_torch=True),
        "val_auc": evaluate_auc(fnn_model, x_va, y_va, is_torch=True),
        "test_auc": evaluate_auc(fnn_model, x_te, y_te, is_torch=True),
    }
    save_params(fnn_model, out_dir, "fnn", is_torch=True)
    with open(out_dir / "fnn_metrics.json", "w") as f:
        json.dump(fnn_res, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare logistic regression and FCNN on simulation data")
    parser.add_argument("--data-dir", default="./data/prostate", help="Directory containing simulation dataset")
    parser.add_argument("--output-dir", default="experiment/comparison", help="Where to store results")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
