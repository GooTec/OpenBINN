#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_original.py
Perform hyper-parameter search and final training for the original
simulation datasets. The dataset location is determined by ``--beta``
and ``--gamma`` (e.g. ``data/b2_g1.5``). The resulting optimal
parameters will be used by ``train_variants.py`` to train all variant
datasets.
"""

import os
import time
import warnings
import argparse
from pathlib import Path
from itertools import product

from importance_calculation import (
    explain_dataset as explain_orig_dataset,
    METHOD as IMP_METHOD,
)

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader as GeoLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score
)

from openxai.binn import PNet
from openxai.binn.util import InMemoryLogger, get_roc
from openxai.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

# ───────────────────────────────────
LR_LIST     = [1e-3, 5e-3]
BS_LIST     = [8, 16]
MAX_EPOCHS  = 200
PATIENCE    = 10
N_SIM       = 100
N_VARIANTS  = 100
DEFAULT_BETA  = 2
DEFAULT_GAMMA = 2
DATA_ROOT   = Path(f"./data/b{DEFAULT_BETA}_g{DEFAULT_GAMMA}")
NUM_WORKERS = 0
# ───────────────────────────────────

def bin_stats(y, p):
    pred = p[:, 1] > 0.5
    return dict(
        acc=accuracy_score(y, pred),
        aupr=average_precision_score(y, p[:, 1]),
        f1=f1_score(y, pred),
        prec=precision_score(y, pred),
        rec=recall_score(y, pred)
    )

def save_roc(fpr, tpr, auc_val, title, path):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUROC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(frameon=False); plt.tight_layout()
    plt.savefig(path); plt.close(fig)

def load_reactome_once():
    return ReactomeNetwork(dict(
        reactome_base_dir="../biological_knowledge/simulation",
        relations_file_name="SimulationPathwaysRelation.txt",
        pathway_names_file_name="SimulationPathways.txt",
        pathway_genes_file_name="SimulationPathways.gmt",
    ))

# ╭──────────────────────────────────────────────────────────────╮
# │  train_dataset():                                            │
# │  · best_params = None  → Grid search + 최적 학습 + 반환      │
# │  · best_params = (lr*, bs*) → 그 값으로만 학습·평가          │
# ╰──────────────────────────────────────────────────────────────╯
def load_best_params(metrics_fp: Path):
    if metrics_fp.exists():
        df = pd.read_csv(metrics_fp)
        lr = float(df.iloc[0].get("best_lr", df.iloc[0].get("best_lr", 1e-3)))
        bs_col = "best_bs" if "best_bs" in df.columns else "best_batch"
        bs = int(df.iloc[0].get(bs_col, 16))
        return (lr, bs)
    return None


def train_dataset(scen_dir: Path, reactome, best_params=None):
    if not (scen_dir / "splits").exists():
        print(f"[WARN] splits 없음: {scen_dir}")
        return

    ds = PnetSimDataSet(root=str(scen_dir), num_features=3)
    ds.split_index_by_file(
        train_fp=scen_dir/"splits"/"training_set_0.csv",
        valid_fp=scen_dir/"splits"/"validation_set.csv",
        test_fp =scen_dir/"splits"/"test_set.csv",
    )

    maps = get_layer_maps(
        genes=list(ds.node_index), reactome=reactome,
        n_levels=3, direction="root_to_leaf", add_unk_genes=False
    )
    ds.node_index = [g for g in ds.node_index if g in maps[0].index]

    results_root = scen_dir/"results"
    search_root  = results_root/"hparam_search"
    search_root.mkdir(parents=True, exist_ok=True)
    metrics_fp   = results_root/"optimal"/"metrics.csv"

    if metrics_fp.exists() and best_params is None:
        print(f"[skip] already trained → {scen_dir.relative_to(DATA_ROOT.parent)}")
        best_params = load_best_params(metrics_fp)
    

    # ────────────────────────────────
    # ① Grid-Search (original 데이터)
    # ────────────────────────────────
    if best_params is None:
        loaded = load_best_params(metrics_fp)
        if loaded is not None:
            best_params = loaded

    if best_params is None:
        summary_rows = []
        for lr, bs in product(LR_LIST, BS_LIST):
            tag = f"lr_{lr:g}_bs_{bs}"
            print(f"      Grid ▶ {scen_dir.relative_to(DATA_ROOT.parent)} | {tag}")

            tr_loader = GeoLoader(ds, bs,
                                  sampler=SubsetRandomSampler(ds.train_idx),
                                  num_workers=NUM_WORKERS)
            va_loader = GeoLoader(ds, bs,
                                  sampler=SubsetRandomSampler(ds.valid_idx),
                                  num_workers=NUM_WORKERS)

            model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=lr)
            trainer = pl.Trainer(
                accelerator="auto", deterministic=True, max_epochs=MAX_EPOCHS,
                callbacks=[EarlyStopping("val_loss", patience=PATIENCE,
                                         mode="min", verbose=False, min_delta=0.01)],
                logger=InMemoryLogger(), enable_progress_bar=False
            )
            t0 = time.time(); trainer.fit(model, tr_loader, va_loader)
            run_t = time.time() - t0

            _, _, tr_auc, _, _ = get_roc(model, tr_loader, exp=False)
            fpr, tpr, va_auc, yv, pv = get_roc(model, va_loader, exp=False)

            exp_dir = search_root/tag; exp_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), exp_dir/"trained_model.pth")
            save_roc(fpr, tpr, va_auc, "Validation ROC", exp_dir/"roc_valid.png")

            summary_rows.append({
                "lr": lr, "bs": bs, "val_auc": va_auc,
                "runtime": run_t, "train_auc": tr_auc,
                **{f"val_{k}": v for k,v in bin_stats(yv,pv).items()}
            })

        df = pd.DataFrame(summary_rows).sort_values("val_auc", ascending=False)
        best_lr, best_bs = df.iloc[0][["lr","bs"]].tolist()
        best_params = (best_lr, int(best_bs))
    else:
        best_lr, best_bs = best_params

    # ────────────────────────────────
    # ② 고정 파라미터로 학습 & 평가
    # ────────────────────────────────
    tr_loader = GeoLoader(ds, best_bs,
                          sampler=SubsetRandomSampler(ds.train_idx),
                          num_workers=NUM_WORKERS)
    va_loader = GeoLoader(ds, best_bs,
                          sampler=SubsetRandomSampler(ds.valid_idx),
                          num_workers=NUM_WORKERS)
    te_loader = GeoLoader(ds, best_bs,
                          sampler=SubsetRandomSampler(ds.test_idx),
                          num_workers=NUM_WORKERS)

    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=best_lr)
    trainer = pl.Trainer(
        accelerator="auto", deterministic=True, max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping("val_loss", patience=PATIENCE, mode="min", verbose=False, min_delta=0.01)],
        logger=InMemoryLogger(), enable_progress_bar=False
    )
    trainer.fit(model, tr_loader, va_loader)

    _, _, tr_auc, _, _ = get_roc(model, tr_loader, exp=False)
    fv, tv, va_auc, yv, pv = get_roc(model, va_loader, exp=False)
    ft, tt, te_auc, yt, pt = get_roc(model, te_loader, exp=False)

    opt_dir = results_root/"optimal"; opt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), opt_dir/"trained_model.pth")
    save_roc(fv,tv,va_auc,"Validation ROC", opt_dir/"roc_valid.png")
    save_roc(ft,tt,te_auc,"Test ROC",       opt_dir/"roc_test.png")

    pd.DataFrame([
        {
            "best_lr": best_lr,
            "best_bs": best_bs,
            "train_auc": tr_auc,
            "val_auc": va_auc,
            "test_auc": te_auc,
            **{f"val_{k}": v for k, v in bin_stats(yv, pv).items()},
            **{f"test_{k}": v for k, v in bin_stats(yt, pt).items()},
        }
    ]).to_csv(opt_dir/"metrics.csv", index=False)

    imp_fp = scen_dir / f"PNet_{IMP_METHOD}_target_scores.csv"
    if not imp_fp.exists():
        explain_orig_dataset(scen_dir, reactome)

    return best_params  # (lr*, bs*)

# ╭──────────────────────────────────────────────────────────────╮
# │                           main()                             │
# ╰──────────────────────────────────────────────────────────────╯
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    pl.seed_everything(42, workers=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim",   type=int, default=N_SIM)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    args = ap.parse_args()

    data_root = Path(f"./data/b{args.beta}_g{args.gamma}")

    reactome = load_reactome_once()

    for i in range(args.start_sim, args.end_sim + 1):
        base_dir = data_root / f"{i}"
        print(f"\n■■ Simulation {i:3d} ■■")

        train_dataset(base_dir, reactome, best_params=None)

    print("\n✓ original datasets trained.")

if __name__ == "__main__":
    main()
