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
from openbinn.binn.util import (
    InMemoryLogger,
    get_roc,
    MetricsRecorder,
    eval_metrics,
    GradNormPrinter,
)
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
import json


PATHWAY_LINEAR_EFFECT = 2.0
PATHWAY_NONLINEAR_EFFECT = 2.0
METHODS = ["ig", "shap", "deeplift", "deepliftshap"]
NO_BASELINE_METHODS = {"sg", "grad", "gradshap", "lrp", "lime", "control", "feature_ablation"}

class ModelWrapper(torch.nn.Module):
    def __init__(self, model: PNet, target_layer: int):
        super().__init__()
        self.model = model
        self.print_layer = target_layer
        self.target_layer = target_layer

    def forward(self, x):
        outs = self.model(x)
        return outs[self.target_layer - 1]

def generate(nonlinear: bool = True) -> None:
    cmd = [
        "python",
        "generate_simulations.py",
        "--pathway_linear_effect", str(PATHWAY_LINEAR_EFFECT),
        "--pathway_nonlinear_effect", str(PATHWAY_NONLINEAR_EFFECT),
        "--n_sim", "1",
        "--exp", str(EXP_NUM),
    ]
    if nonlinear:
        cmd.append("--pathway_nonlinear")
    subprocess.run(cmd, check=True)

def load_reactome_once():
    return ReactomeNetwork(
        dict(
            reactome_base_dir="../biological_knowledge/simulation",
            relations_file_name="SimulationPathwaysRelation.txt",
            pathway_names_file_name="SimulationPathways.txt",
            pathway_genes_file_name="SimulationPathways.gmt",
        )
    )

def train_dataset(
    data_dir: Path,
    results_dir: Path,
    reactome,
    lr: float,
    batch_size: int,
    args,
):
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
    ds.align_with_map(maps[0].index)

    class PerfCallback(MetricsRecorder):
        def __init__(self, out_dir: Path, tr_loader, va_loader, te_loader, period: int = 10):
            super().__init__(out_dir, tr_loader, va_loader, te_loader, period)

    run_dir = results_dir
    tr_loader = GeoLoader(
        ds,
        batch_size,
        sampler=SubsetRandomSampler(ds.train_idx),
        num_workers=0,
    )
    va_loader = GeoLoader(
        ds,
        batch_size,
        sampler=SubsetRandomSampler(ds.valid_idx),
        num_workers=0,
    )
    te_loader = GeoLoader(
        ds,
        batch_size,
        sampler=SubsetRandomSampler(ds.test_idx),
        num_workers=0,
    )

    callback = PerfCallback(run_dir, tr_loader, va_loader, te_loader, period=10)
    mc = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor=args.monitor_metric,
        mode="max" if args.monitor_metric == "val_auc" else "min",
        save_top_k=1,
        every_n_epochs=10,
        save_last=True,
    )

    optim_cfg = {
        "opt": args.optimizer,
        "lr": lr,
        "wd": args.weight_decay,
        "scheduler": args.scheduler,
        "monitor": args.monitor_metric,
    }

    loss_cfg = {
        "main": args.main_loss_weight,
        "aux": args.aux_loss_weight,
    }
    if args.per_layer_weights:
        loss_cfg["per_layer"] = [float(x) for x in args.per_layer_weights.split(",")]

    model = PNet(
        layers=maps,
        num_genes=maps[0].shape[0],
        lr=lr,
        norm_type=args.norm_type,
        dropout_rate=args.dropout_rate,
        input_dropout=args.input_dropout,
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
    )

    init_loss, init_acc, init_auc = eval_metrics(model, va_loader)
    print(
        f"      Start: loss={init_loss:.4f} acc={init_acc:.4f} auc={init_auc:.4f}"
    )

    trainer = pl.Trainer(
        accelerator="auto",
        deterministic=True,
        max_epochs=200,
        callbacks=[
            pl.callbacks.EarlyStopping(
                args.monitor_metric,
                patience=30,
                mode="max" if args.monitor_metric == "val_auc" else "min",
                verbose=False,
                min_delta=0.01,
            ),
            callback,
            GradNormPrinter(),
            mc,
        ],
        logger=InMemoryLogger(),
        enable_progress_bar=False,
        gradient_clip_val=args.grad_clip_val,
    )

    if args.auto_lr_find or args.auto_scale_bs:
        tuner = pl.tuner.Tuner(trainer)
        if args.auto_lr_find:
            lr_finder = tuner.lr_find(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
            lr = lr_finder.suggestion()
            model.lr = lr
        if args.auto_scale_bs:
            new_bs = tuner.scale_batch_size(model, train_dataloaders=tr_loader, val_dataloaders=va_loader, init_val=batch_size)
            batch_size = new_bs
            tr_loader = GeoLoader(ds, batch_size, sampler=SubsetRandomSampler(ds.train_idx), num_workers=0)
            va_loader = GeoLoader(ds, batch_size, sampler=SubsetRandomSampler(ds.valid_idx), num_workers=0)
            te_loader = GeoLoader(ds, batch_size, sampler=SubsetRandomSampler(ds.test_idx), num_workers=0)

    trainer.fit(model, tr_loader, va_loader)
    fin_loss, fin_acc, fin_auc = eval_metrics(model, va_loader)
    print(
        f"      End  : loss={fin_loss:.4f} acc={fin_acc:.4f} auc={fin_auc:.4f}"
    )

    state = torch.load(mc.best_model_path, map_location="cpu") if mc.best_model_path else model.state_dict()
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    (results_dir / "optimal").mkdir(parents=True, exist_ok=True)
    torch.save(state, results_dir / "optimal" / "trained_model.pth")

    return maps, fin_loss, fin_auc, {
        "learning_rate": lr,
        "batch_size": batch_size,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "norm_type": args.norm_type,
        "dropout_rate": args.dropout_rate,
        "loss_cfg": loss_cfg,
    }, state

def explain_dataset(data_dir: Path, results_dir: Path, reactome, maps, args=None):
    ds = PnetSimExpDataSet(root=str(data_dir), num_features=1)
    ds.split_index_by_file(
        train_fp=data_dir / 'splits' / 'training_set_0.csv',
        valid_fp=data_dir / 'splits' / 'validation_set.csv',
        test_fp=data_dir / 'splits' / 'test_set.csv',
    )
    maps = get_layer_maps(
        genes=list(ds.node_index),
        reactome=reactome,
        n_levels=3,
        direction='root_to_leaf',
        add_unk_genes=False
    )
    ds.align_with_map(maps[0].index)
    model = PNet(
        layers=maps,
        num_genes=maps[0].shape[0],
        lr=0.001,
        norm_type=args.norm_type if args else "batchnorm",
        dropout_rate=args.dropout_rate if args else 0.1,
        input_dropout=args.input_dropout if args else 0.5,
    )
    state = torch.load(results_dir / 'optimal' / 'trained_model.pth', map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    loader = GeoLoader(ds, batch_size=len(ds.test_idx), sampler=SubsetRandomSampler(ds.test_idx), num_workers=0)
    explain_root = results_dir / 'explanations' / 'PNET'
    explain_root.mkdir(parents=True, exist_ok=True)
    data_label = data_dir.name
    config = utils.load_config("../configs/experiment_config.json")

    print("#samples in ds         :", len(ds))
    print("#test-indices in ds    :", len(ds.test_idx))
    print("first 10 test-idx      :", ds.test_idx[:10])

    methods = [m for m in METHODS if (m == "deepliftshap" and "shap" in config['explainers']) or m in config['explainers']]
    for method in methods:
        base_method = "shap" if method == "deepliftshap" else method
        for tgt in range(1, len(maps)+1):
            print(f"Explaining {method} for target layer {tgt} ...")
            wrap = ModelWrapper(model, tgt)
            expl_acc, lab_acc, pred_acc, id_acc = {}, [], [], []
            for X, y, ids in loader:
                p_conf = utils.fill_param_dict(base_method, config['explainers'][base_method], X)
                p_conf['classification_type'] = 'binary'
                if base_method not in NO_BASELINE_METHODS:
                    p_conf['baseline'] = torch.zeros_like(X)
                explainer = Explainer(base_method, wrap, p_conf)
                exp_dict = explainer.get_layer_explanations(X, y)
                for lname, ten in exp_dict.items():
                    expl_acc.setdefault(lname, []).append(ten.detach().cpu().numpy())
                lab_acc.append(y.cpu().numpy())
                pred_acc.append(wrap(X).detach().cpu().numpy())
                id_acc.append(ids)

            print(len(expl_acc), "explanations found for", method)
            for idx, (lname, arrs) in enumerate(expl_acc.items()):
                print(f"Saving {method} layer {tgt} explanation for {lname} ...")
                arr = np.concatenate(arrs, axis=0)
                labels = np.concatenate(lab_acc, axis=0)
                preds = np.concatenate(pred_acc, axis=0)
                all_ids = [sid for batch in id_acc for sid in batch]

                cols = None
                for m in maps:
                    if arr.shape[1] == m.shape[0]:
                        cols = list(m.index)
                        break
                    if arr.shape[1] == m.shape[1]:
                        cols = list(m.columns)
                        break
                if cols is None:
                    cols = [f"f{i}" for i in range(arr.shape[1])]

                df = pd.DataFrame(arr, columns=cols)
                df['label'] = labels
                df['prediction'] = preds
                df['sample_id'] = all_ids
                out_fp = explain_root / f"PNet_{data_label}_{method}_L{tgt}_layer{idx}_test.csv"
                df.to_csv(out_fp, index=False)
        print(f"Saved raw importances for {method}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run experiment 1")
    ap.add_argument("--exp", type=int, default=1, help="Experiment number")
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    ap.add_argument(
        "--scheduler", choices=["none", "cosine", "plateau", "onecycle"], default="none"
    )
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--monitor_metric", choices=["val_loss", "val_auc"], default="val_loss")
    ap.add_argument("--norm_type", choices=["batchnorm", "layernorm", "none"], default="batchnorm")
    ap.add_argument("--dropout_rate", type=float, default=0.1)
    ap.add_argument("--input_dropout", type=float, default=0.5)
    ap.add_argument("--main_loss_weight", type=float, default=1.0)
    ap.add_argument("--aux_loss_weight", type=float, default=0.0)
    ap.add_argument("--per_layer_weights", type=str, default=None)
    ap.add_argument("--auto_lr_find", action="store_true")
    ap.add_argument("--auto_scale_bs", action="store_true")
    ap.add_argument("--lr_list", type=float, nargs="*")
    ap.add_argument("--bs_list", type=int, nargs="*")
    ap.add_argument("--grad_clip_val", type=float, default=0.0)
    args = ap.parse_args()
    EXP_NUM = args.exp

    reactome = load_reactome_once()
    generate(nonlinear=True)
    scenario_id = Path(f"b{PATHWAY_LINEAR_EFFECT}_g{PATHWAY_NONLINEAR_EFFECT}") / "1"
    data_dir = Path("data") / f"experiment{EXP_NUM}" / scenario_id
    results_dir = Path("results") / f"experiment{EXP_NUM}" / scenario_id
    if args.lr_list or args.bs_list:
        lrs = args.lr_list or [args.learning_rate]
        bss = args.bs_list or [args.batch_size]
        summary = []
        best = None
        best_maps = None
        best_cfg = None
        best_state = None
        for lr in lrs:
            for bs in bss:
                maps, vloss, vauc, cfg, state = train_dataset(data_dir, results_dir, reactome, lr, bs, args)
                summary.append({"lr": lr, "batch_size": bs, "val_loss": vloss, "val_auc": vauc})
                score = vauc if args.monitor_metric == "val_auc" else vloss
                better = (best is None) or (
                    score > best if args.monitor_metric == "val_auc" else score < best
                )
                if better:
                    best = score
                    best_maps = maps
                    best_cfg = cfg
                    best_state = state
        pd.DataFrame(summary).to_csv(results_dir / "summary.csv", index=False)
        maps = best_maps
        (results_dir / "optimal").mkdir(parents=True, exist_ok=True)
        torch.save(best_state, results_dir / "optimal" / "trained_model.pth")
        with open(results_dir / "optimal" / "best_params.json", "w") as f:
            json.dump(best_cfg, f)
    else:
        maps, _, _, cfg, state = train_dataset(
            data_dir, results_dir, reactome, args.learning_rate, args.batch_size, args
        )
        with open(results_dir / "optimal" / "best_params.json", "w") as f:
            json.dump(cfg, f)

    explain_dataset(data_dir, results_dir, reactome, maps, args)
