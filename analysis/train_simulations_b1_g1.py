#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_all_variants_fast.py
──────────────────────────
• b4_g4 시뮬레이션(1‒100)  →  원본 데이터에 대해 한 번 Grid-Search
  → best (lr*, bs*) 도출  → 그 값으로 변형 300 세트 학습.
"""

import os, time, warnings, argparse
from pathlib import Path
from itertools import product
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from openbinn.binn import PNet
from openbinn.binn.util import InMemoryLogger, get_roc
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

# ───────────────────────────────────
LR_LIST     = [1e-3, 5e-3]
BS_LIST     = [8, 16]
MAX_EPOCHS  = 200
PATIENCE    = 10
N_SIM       = 100
N_VARIANTS  = 100
DATA_ROOT   = Path("./data/b4_g4")
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

    # ────────────────────────────────
    # ① Grid-Search (original 에서만)
    # ────────────────────────────────
    # if best_params is None:
    #     summary_rows = []
    #     for lr, bs in product(LR_LIST, BS_LIST):
    #         tag = f"lr_{lr:g}_bs_{bs}"
    #         print(f"      Grid ▶ {scen_dir.relative_to(DATA_ROOT)} | {tag}")

    #         tr_loader = GeoLoader(ds, bs,
    #                               sampler=SubsetRandomSampler(ds.train_idx),
    #                               num_workers=NUM_WORKERS)
    #         va_loader = GeoLoader(ds, bs,
    #                               sampler=SubsetRandomSampler(ds.valid_idx),
    #                               num_workers=NUM_WORKERS)

    #         model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=lr)
    #         trainer = pl.Trainer(
    #             accelerator="auto", deterministic=True, max_epochs=MAX_EPOCHS,
    #             callbacks=[EarlyStopping("val_loss", patience=PATIENCE,
    #                                      mode="min", verbose=False, min_delta=0.01)],
    #             logger=InMemoryLogger(), enable_progress_bar=False
    #         )
    #         t0 = time.time(); trainer.fit(model, tr_loader, va_loader)
    #         run_t = time.time() - t0

    #         _, _, tr_auc, _, _ = get_roc(model, tr_loader, exp=False)
    #         fpr, tpr, va_auc, yv, pv = get_roc(model, va_loader, exp=False)

    #         exp_dir = search_root/tag; exp_dir.mkdir(exist_ok=True)
    #         torch.save(model.state_dict(), exp_dir/"trained_model.pth")
    #         save_roc(fpr, tpr, va_auc, "Validation ROC", exp_dir/"roc_valid.png")

    #         summary_rows.append({
    #             "lr": lr, "bs": bs, "val_auc": va_auc,
    #             "runtime": run_t, "train_auc": tr_auc,
    #             **{f"val_{k}": v for k,v in bin_stats(yv,pv).items()}
    #         })

    #     df = pd.DataFrame(summary_rows).sort_values("val_auc", ascending=False)
    #     best_lr, best_bs = df.iloc[0][["lr","bs"]].tolist()
    #     best_params = (best_lr, int(best_bs))
    # # ────────────────────────────────

    best_lr, best_bs = 1e-3, 16

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

    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=best_lr,
                 diversity_lambda=0.1)
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

    pd.DataFrame([{
        "best_lr": best_lr, "best_bs": best_bs,
        "train_auc": tr_auc, "val_auc": va_auc, "test_auc": te_auc,
        **{f"val_{k}": v for k,v in bin_stats(yv,pv).items()},
        **{f"test_{k}": v for k,v in bin_stats(yt,pt).items()}
    }]).to_csv(opt_dir/"metrics.csv", index=False)

    return best_params        # (lr*, bs*)

# ╭──────────────────────────────────────────────────────────────╮
# │                           main()                             │
# ╰──────────────────────────────────────────────────────────────╯
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    pl.seed_everything(42, workers=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim",   type=int, default=N_SIM)
    args = ap.parse_args()

    reactome = load_reactome_once()

    for i in range(args.start_sim, args.end_sim+1):
        base_dir = DATA_ROOT/f"{i}"
        print(f"\n■■ Simulation {i:3d} ■■")

        # ① original → grid search
        best_params = train_dataset(base_dir, reactome, best_params=None)

        # ② variants → 고정 best_params
        for vtype in ["gene-permutation"]:
            for b in range(1, N_VARIANTS+1):
                v_dir = base_dir/vtype/f"{b}"
                if v_dir.exists():
                    print(f"    Variant ▶ {vtype}/{b}")
                    train_dataset(v_dir, reactome, best_params=best_params)

    print("\n✓ 모든 데이터셋 학습 완료.")

if __name__ == "__main__":
    main()
