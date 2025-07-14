import sys
from pathlib import Path
cwd = Path.cwd()
if (cwd / 'openbinn').exists():
    sys.path.insert(0, str(cwd))
elif (cwd.parent / 'openbinn').exists():
    sys.path.insert(0, str(cwd.parent))

import argparse

import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader as GeoLoader

from openbinn.binn import PNet
from openbinn.binn.util import InMemoryLogger, get_roc
from openbinn.binn.data import (
    PnetSimDataSet,
    PnetSimExpDataSet,
    ReactomeNetwork,
    get_layer_maps,
)
from openbinn.explainer import Explainer
import openbinn.experiment_utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BETA_LIST = [0.5, 1.0, 2.0, 4.0]
GAMMA_LIST = [0.5, 1.0, 2.0, 4.0]
METHODS = ["deeplift", "ig", "gradshap", "itg", "shap"]

class ModelWrapper(torch.nn.Module):
    def __init__(self, model: PNet, target_layer: int):
        super().__init__()
        self.model = model
        self.print_layer = target_layer
        self.target_layer = target_layer

    def forward(self, x):
        outs = self.model(x)
        return outs[self.target_layer - 1]

def generate(beta: float, gamma: float) -> None:
    subprocess.run(
        [
            "python",
            "generate_simulations.py",
            "--beta", str(beta),
            "--gamma", str(gamma),
            "--n_sim", "1",
            "--exp", str(EXP_NUM),
        ],
        check=True,
    )

def load_reactome_once():
    return ReactomeNetwork(
        dict(
            reactome_base_dir="../biological_knowledge/simulation",
            relations_file_name="SimulationPathwaysRelation.txt",
            pathway_names_file_name="SimulationPathways.txt",
            pathway_genes_file_name="SimulationPathways.gmt",
        )
    )

def train_dataset(data_dir: Path, results_dir: Path, reactome):
    ds = PnetSimDataSet(root=str(data_dir), num_features=3)
    ds.split_index_by_file(
        train_fp=data_dir / "splits" / "training_set_0.csv",
        valid_fp=data_dir / "splits" / "validation_set.csv",
        test_fp=data_dir / "splits" / "test_set.csv",
    )
    maps = get_layer_maps(
        genes=list(ds.node_index),
        reactome=reactome,
        n_levels=3,
        direction="root_to_leaf",
        add_unk_genes=False,
    )
    ds.node_index = [g for g in ds.node_index if g in maps[0].index]
    tr_loader = GeoLoader(ds, 16, sampler=SubsetRandomSampler(ds.train_idx), num_workers=0)
    va_loader = GeoLoader(ds, 16, sampler=SubsetRandomSampler(ds.valid_idx), num_workers=0)
    te_loader = GeoLoader(ds, 16, sampler=SubsetRandomSampler(ds.test_idx), num_workers=0)

    class PerfCallback(pl.Callback):
        def __init__(self, period: int = 10):
            self.period = period
            self.records = []

        def _compute_loss_acc(self, pl_module, loader):
            device = pl_module.device
            total_loss = 0.0
            correct = 0.0
            total = 0
            pl_module.eval()
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)
                    out = pl_module.step((x, y), "eval")
                    total_loss += out["loss"].item() * out["total"]
                    correct += out["correct"].item()
                    total += out["total"]
            return total_loss / total, correct / total

        def on_validation_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            if epoch % self.period == 0 or epoch == trainer.max_epochs:
                tr_auc = get_roc(pl_module, tr_loader, exp=False)[2]
                va_auc = get_roc(pl_module, va_loader, exp=False)[2]
                te_auc = get_roc(pl_module, te_loader, exp=False)[2]
                tr_loss, tr_acc = self._compute_loss_acc(pl_module, tr_loader)
                va_loss, va_acc = self._compute_loss_acc(pl_module, va_loader)
                te_loss, te_acc = self._compute_loss_acc(pl_module, te_loader)
                self.records.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(tr_loss),
                        "val_loss": float(va_loss),
                        "test_loss": float(te_loss),
                        "train_accuracy": float(tr_acc),
                        "val_accuracy": float(va_acc),
                        "test_accuracy": float(te_acc),
                        "train_auc": float(tr_auc),
                        "val_auc": float(va_auc),
                        "test_auc": float(te_auc),
                    }
                )
                df = pd.DataFrame(self.records)
                perf_dir = results_dir / "performance"
                perf_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(perf_dir / "metrics.csv", index=False)
                vis_dir = results_dir / "visualize"
                vis_dir.mkdir(parents=True, exist_ok=True)
                plt.figure()
                plt.plot(df["epoch"], df["train_loss"], label="train")
                plt.plot(df["epoch"], df["val_loss"], label="val")
                plt.plot(df["epoch"], df["test_loss"], label="test")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(vis_dir / "loss_curve.png")
                plt.close()

                plt.figure()
                plt.plot(df["epoch"], df["train_accuracy"], label="train")
                plt.plot(df["epoch"], df["val_accuracy"], label="val")
                plt.plot(df["epoch"], df["test_accuracy"], label="test")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.tight_layout()
                plt.savefig(vis_dir / "accuracy_curve.png")
                plt.close()

                plt.figure()
                plt.plot(df["epoch"], df["train_auc"], label="train")
                plt.plot(df["epoch"], df["val_auc"], label="val")
                plt.plot(df["epoch"], df["test_auc"], label="test")
                plt.xlabel("Epoch")
                plt.ylabel("AUC")
                plt.legend()
                plt.tight_layout()
                plt.savefig(vis_dir / "auc_curve.png")
                plt.close()

    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=1e-3)
    trainer = pl.Trainer(
        accelerator="auto",
        deterministic=True,
        max_epochs=200,
        callbacks=[
            pl.callbacks.EarlyStopping("val_loss", patience=10, mode="min", verbose=False, min_delta=0.01),
            PerfCallback(period=10),
            ModelCheckpoint(dirpath=results_dir/"checkpoints", monitor="val_loss", mode="min", save_top_k=1, every_n_epochs=10, save_last=True),
        ],
        logger=InMemoryLogger(),
        enable_progress_bar=False,
    )
    trainer.fit(model, tr_loader, va_loader)
    (results_dir / "optimal").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), results_dir / "optimal" / "trained_model.pth")
    return maps

def explain_dataset(data_dir: Path, results_dir: Path, reactome, maps, method: str):
    ds = PnetSimExpDataSet(root=str(data_dir), num_features=1)
    ds.split_index_by_file(
        train_fp=data_dir / 'splits' / 'training_set_0.csv',
        valid_fp=data_dir / 'splits' / 'validation_set.csv',
        test_fp = data_dir / 'splits' / 'test_set.csv',
    )
    maps = get_layer_maps(
        genes=list(ds.node_index),
        reactome=reactome,
        n_levels=3,
        direction='root_to_leaf',
        add_unk_genes=False
    )
    ds.node_index = [g for g in ds.node_index if g in maps[0].index]
    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=0.001)
    state = torch.load(results_dir / 'optimal' / 'trained_model.pth', map_location='cpu')
    model.load_state_dict(state); model.eval()
    loader = GeoLoader(ds, batch_size=len(ds.test_idx), sampler=SubsetRandomSampler(ds.test_idx), num_workers=0)
    explain_root = results_dir / 'explanations'
    explain_root.mkdir(exist_ok=True)

    print("#samples in ds         :", len(ds))
    print("#test-indices in ds    :", len(ds.test_idx))
    print("first 10 test-idx      :", ds.test_idx[:10])

    for tgt in range(1, len(maps)+1):
        print(f"Explaining {method} for target layer {tgt} ...")
        wrap = ModelWrapper(model, tgt)
        expl_acc, lab_acc, pred_acc, id_acc = {}, [], [], []
        for X, y, ids in loader:
            p_conf = {'baseline': torch.zeros_like(X), 'classification_type': 'binary'}
            explainer = Explainer(method, wrap, p_conf)
            exp_dict = explainer.get_layer_explanations(X, y)
            for lname, ten in exp_dict.items():
                expl_acc.setdefault(lname, []).append(ten.detach().cpu().numpy())
            lab_acc.append(y.cpu().numpy())
            pred_acc.append(wrap(X).detach().cpu().numpy())
            id_acc.append(ids)

        print(len(expl_acc), "explanations found for", method)
        for idx, (lname, arrs) in enumerate(expl_acc.items()):
            print(f"Saving {method} layer {tgt} explanation for {lname} ...")
            if idx >= len(maps):
                break
            arr = np.concatenate(arrs, axis=0)
            labels = np.concatenate(lab_acc, axis=0)
            preds  = np.concatenate(pred_acc, axis=0)
            all_ids= [sid for batch in id_acc for sid in batch]
            cur_map = maps[idx]
            cols = list(cur_map.index) if cur_map.shape[0]==arr.shape[1] else list(cur_map.columns)
            df = pd.DataFrame(arr, columns=cols)
            df['label'] = labels
            df['prediction'] = preds
            df['sample_id'] = all_ids
            out_fp = explain_root / f"PNet_{method}_L{tgt}_layer{idx}_test.csv"
            df.to_csv(out_fp, index=False)
    print(f"Saved raw importances for {method}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run experiment 1")
    ap.add_argument("--exp", type=int, default=1, help="Experiment number")
    args = ap.parse_args()
    EXP_NUM = args.exp

    reactome = load_reactome_once()
    for beta in BETA_LIST:
        for gamma in GAMMA_LIST:
            generate(beta, gamma)
            scenario_id = Path(f"b{beta}_g{gamma}") / "1"
            data_dir = Path("data") / f"experiment{EXP_NUM}" / scenario_id
            results_dir = Path("results") / f"experiment{EXP_NUM}" / scenario_id
            maps = train_dataset(data_dir, results_dir, reactome)
            for method in METHODS:
                explain_dataset(data_dir, results_dir, reactome, maps, method)
