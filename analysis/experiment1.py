import subprocess
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader as GeoLoader

from openbinn.binn import PNet
from openbinn.binn.util import InMemoryLogger
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


BETA_LIST = [0.0, 0.5, 1.0, 1.5, 2.0]
GAMMA_LIST = [0.0, 0.5, 1.0, 2.0]
METHODS = ["deeplift", "ig", "gradshap", "itg", "shap"]


def generate(beta: float, gamma: float):
    subprocess.run(
        [
            "python",
            "analysis/generate_simulations.py",
            "--beta",
            str(beta),
            "--gamma",
            str(gamma),
            "--n_sim",
            "1",
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


def train_dataset(scen_dir: Path, reactome):
    ds = PnetSimDataSet(root=str(scen_dir), num_features=3)
    ds.split_index_by_file(
        train_fp=scen_dir / "splits" / "training_set_0.csv",
        valid_fp=scen_dir / "splits" / "validation_set.csv",
        test_fp=scen_dir / "splits" / "test_set.csv",
    )
    maps = get_layer_maps(
        genes=list(ds.node_index),
        reactome=reactome,
        n_levels=3,
        direction="root_to_leaf",
        add_unk_genes=False,
    )
    ds.node_index = [g for g in ds.node_index if g in maps[0].index]
    tr_loader = GeoLoader(
        ds,
        16,
        sampler=SubsetRandomSampler(ds.train_idx),
        num_workers=0,
    )
    va_loader = GeoLoader(
        ds,
        16,
        sampler=SubsetRandomSampler(ds.valid_idx),
        num_workers=0,
    )
    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=1e-3)
    trainer = pl.Trainer(
        accelerator="auto",
        deterministic=True,
        max_epochs=200,
        callbacks=[pl.callbacks.EarlyStopping("val_loss", patience=10, mode="min", verbose=False, min_delta=0.01)],
        logger=InMemoryLogger(),
        enable_progress_bar=False,
    )
    trainer.fit(model, tr_loader, va_loader)
    (scen_dir / "results" / "optimal").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), scen_dir / "results" / "optimal" / "trained_model.pth")
    return maps


def explain_dataset(scen_dir: Path, reactome, maps, method: str):
    ds = PnetSimExpDataSet(root=str(scen_dir), num_features=1)
    ds.split_index_by_file(
        train_fp=scen_dir / "splits" / "training_set_0.csv",
        valid_fp=scen_dir / "splits" / "validation_set.csv",
        test_fp=scen_dir / "splits" / "test_set.csv",
    )
    maps = get_layer_maps(
        genes=list(ds.node_index),
        reactome=reactome,
        n_levels=3,
        direction="root_to_leaf",
        add_unk_genes=False,
    )
    ds.node_index = [g for g in ds.node_index if g in maps[0].index]
    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=0.001)
    state = torch.load(scen_dir / "results" / "optimal" / "trained_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    loader = GeoLoader(
        ds,
        batch_size=len(ds.test_idx),
        sampler=SubsetRandomSampler(ds.test_idx),
        num_workers=0,
    )
    explain_root = scen_dir / "explanations"
    explain_root.mkdir(exist_ok=True)
    for tgt in range(1, len(maps) + 1):
        wrap = ModelWrapper(model, tgt)
        expl_acc, lab_acc, pred_acc, id_acc = {}, [], [], []
        for X, y, ids in loader:
            p_conf = utils.fill_param_dict(method, {}, X)
            p_conf["classification_type"] = "binary"
            if method not in {"itg", "sg", "grad", "gradshap", "control", "feature_ablation"}:
                p_conf["baseline"] = torch.zeros_like(X)
            explainer = Explainer(method, wrap, p_conf)
            exp_dict = explainer.get_layer_explanations(X, y)
            for lname, ten in exp_dict.items():
                expl_acc.setdefault(lname, []).append(ten.detach().cpu().numpy())
            lab_acc.append(y.cpu().numpy())
            pred_acc.append(wrap(X).detach().cpu().numpy())
            id_acc.append(ids)
        for idx, (lname, arrs) in enumerate(expl_acc.items()):
            if idx >= len(maps):
                break
            arr = np.concatenate(arrs, axis=0)
            labels = np.concatenate(lab_acc, axis=0)
            preds = np.concatenate(pred_acc, axis=0)
            all_ids = [sid for batch in id_acc for sid in batch]
            cur_map = maps[idx]
            cols = list(cur_map.index) if cur_map.shape[0] == arr.shape[1] else list(cur_map.columns)
            df = pd.DataFrame(arr, columns=cols)
            df["label"] = labels
            df["prediction"] = preds
            df["sample_id"] = all_ids
            out_fp = explain_root / f"PNet_{method}_L{tgt}_layer{idx}_test.csv"
            df.to_csv(out_fp, index=False)
    print(f"Saved raw importances for {method}")


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: PNet, target_layer: int):
        super().__init__()
        self.model = model
        self.target_layer = target_layer

    def forward(self, x):
        outs = self.model(x)
        return outs[self.target_layer - 1]


if __name__ == "__main__":
    reactome = load_reactome_once()
    for beta in BETA_LIST:
        for gamma in GAMMA_LIST:
            generate(beta, gamma)
            scenario = Path(f"./data/b{beta}_g{gamma}/1")
            maps = train_dataset(scenario, reactome)
            for method in METHODS:
                explain_dataset(scenario, reactome, maps, method)

