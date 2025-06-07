# -*- coding: utf-8 -*-
"""
2.train_pnet.py – automatic hyper-parameter search for OPENXAI-PNet
===================================================================

* grid search over learning-rate × batch-size
* select best by highest Validation AUROC
* re-train with best hyper-params and evaluate on Test
* stores:
    scenario/results/
        hparam_search/
            lr_{lr}_bs_{bs}/…
        optimal/
            trained_model.pth
            roc_valid.png
            roc_test.png
            metrics.csv
        hparam_search_summary.csv
"""

import argparse, os, time, warnings, math
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader as GeoLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score)

from openbinn.binn import PNet
from openbinn.binn.util import ProgressBar, InMemoryLogger, get_roc
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Grid-search OPENXAI-PNet hyper-parameters and evaluate best")
    ap.add_argument("--scenario", required=True,
                    help="dataset folder, e.g. ./simulation/linear/alpha_1_beta_1")
    ap.add_argument("--lr_grid",
                    default="1e-3,5e-3,1e-2",
                    help="comma-separated learning rates (default: 1e-3,5e-3,1e-2)")
    ap.add_argument("--batch_grid",
                    default="8,16,32",
                    help="comma-separated batch sizes (default: 8,16,32)")
    ap.add_argument("--epochs",   type=int, default=200, help="max epochs")
    ap.add_argument("--patience", type=int, default=20,
                    help="early-stopping patience on val_loss")
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()

# ------------------------------------------------------------------ #
def str2floats(s: str):
    return [float(x) for x in s.split(",")]

def str2ints(s: str):
    return [int(x)   for x in s.split(",")]

def bin_stats(y, p):
    pred = p[:, 1] > 0.5
    return dict(acc=accuracy_score(y, pred),
                aupr=average_precision_score(y, p[:, 1]),
                f1=f1_score(y, pred),
                prec=precision_score(y, pred),
                rec=recall_score(y, pred))

def save_roc(fpr, tpr, auc_val, title, path):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUROC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
    ax.legend(frameon=False); plt.tight_layout(); plt.savefig(path); plt.close(fig)

# ------------------------------------------------------------------ #
def main(cfg: argparse.Namespace):

    # reproducibility
    pl.seed_everything(42, workers=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    # -------------------------------------------------------------- #
    # dataset / splits
    ds = PnetSimDataSet(root=cfg.scenario, num_features=1)
    split_dir = os.path.join(cfg.scenario, "splits")
    ds.split_index_by_file(
        train_fp=os.path.join(split_dir, "training_set_0.csv"),
        valid_fp=os.path.join(split_dir, "validation_set.csv"),
        test_fp =os.path.join(split_dir, "test_set.csv"),
    )

    # reactome hierarchy
    reactome = ReactomeNetwork(dict(
        reactome_base_dir="./biological_knowledge/simulation",
        relations_file_name="SimulationPathwaysRelation.txt",
        pathway_names_file_name="SimulationPathways.txt",
        pathway_genes_file_name="SimulationPathways.gmt",
    ))
    maps = get_layer_maps(
        genes=list(ds.node_index), reactome=reactome,
        n_levels=6, direction="root_to_leaf", add_unk_genes=False)

    # -------------------------------------------------------------- #
    # make results folders
    results_root = os.path.join(cfg.scenario, "results")
    search_root  = os.path.join(results_root, "hparam_search")
    os.makedirs(search_root, exist_ok=True)

    lr_list    = str2floats(cfg.lr_grid)
    bs_list    = str2ints(cfg.batch_grid)

    summary_rows = []

    # -------------------------------------------------------------- #
    # grid search
    for lr, bs in product(lr_list, bs_list):
        tag = f"lr_{lr:g}_bs_{bs}"
        print(f"\n>>> Training {tag}")

        # dataloaders
        train_loader = GeoLoader(ds, bs,
                                 sampler=SubsetRandomSampler(ds.train_idx),
                                 num_workers=cfg.num_workers)
        valid_loader = GeoLoader(ds, bs,
                                 sampler=SubsetRandomSampler(ds.valid_idx),
                                 num_workers=cfg.num_workers)

        # model & trainer
        model   = PNet(layers=maps, num_genes=maps[0].shape[0], lr=lr)
        trainer = pl.Trainer(
            accelerator="auto", deterministic=True,
            max_epochs=cfg.epochs,
            callbacks=[ProgressBar(),
                       EarlyStopping("val_loss", patience=cfg.patience, mode="min")],
            logger=InMemoryLogger(), enable_progress_bar=True)

        t0 = time.time()
        trainer.fit(model, train_loader, valid_loader)
        run_time = time.time() - t0

        # validation metrics
        _, _, tr_auc, _, _            = get_roc(model, train_loader, exp=False)
        fpr_v, tpr_v, va_auc, yv, pv  = get_roc(model, valid_loader, exp=False)
        va_stats = bin_stats(yv, pv)

        # per-experiment save
        exp_dir = os.path.join(search_root, tag)
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "trained_model.pth"))
        save_roc(fpr_v, tpr_v, va_auc, "Validation ROC",
                 os.path.join(exp_dir, "roc_valid.png"))

        # record row
        summary_rows.append({
            "tag":tag, "lr":lr, "batch":bs,
            "epochs_run":trainer.current_epoch+1, "runtime_sec":run_time,
            "train_auc":tr_auc, "val_auc":va_auc, **{f"val_{k}":v for k,v in va_stats.items()}
        })

    # -------------------------------------------------------------- #
    # pick best hyper-parameters
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values("val_auc", ascending=False, inplace=True)
    summary_df.to_csv(os.path.join(results_root, "hparam_search_summary.csv"), index=False)

    best = summary_df.iloc[0]
    best_lr, best_bs = best.lr, int(best.batch)
    print(f"\n*** Best hyper-params → lr={best_lr:g}, batch={best_bs} (val_auc={best.val_auc:.3f}) ***")

    # -------------------------------------------------------------- #
    # re-train with best params & evaluate on test
    train_loader = GeoLoader(ds, best_bs,
                             sampler=SubsetRandomSampler(ds.train_idx),
                             num_workers=cfg.num_workers)
    valid_loader = GeoLoader(ds, best_bs,
                             sampler=SubsetRandomSampler(ds.valid_idx),
                             num_workers=cfg.num_workers)
    test_loader  = GeoLoader(ds, best_bs,
                             sampler=SubsetRandomSampler(ds.test_idx),
                             num_workers=cfg.num_workers)

    best_model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=best_lr)
    trainer_best = pl.Trainer(
        accelerator="auto", deterministic=True,
        max_epochs=cfg.epochs,
        callbacks=[ProgressBar(),
                   EarlyStopping("val_loss", patience=cfg.patience, mode="min")],
        logger=InMemoryLogger(), enable_progress_bar=True)

    trainer_best.fit(best_model, train_loader, valid_loader)

    # metrics
    _, _, tr_auc, _, _           = get_roc(best_model, train_loader, exp=False)
    fpr_v,tpr_v,va_auc,yv,pv     = get_roc(best_model, valid_loader, exp=False)
    fpr_t,tpr_t,te_auc,yt,pt     = get_roc(best_model, test_loader , exp=False)
    va_stats, te_stats = bin_stats(yv, pv), bin_stats(yt, pt)

    # save optimal results
    opt_dir = os.path.join(results_root, "optimal")
    os.makedirs(opt_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(opt_dir, "trained_model.pth"))
    save_roc(fpr_v, tpr_v, va_auc, "Validation ROC", os.path.join(opt_dir, "roc_valid.png"))
    save_roc(fpr_t, tpr_t, te_auc, "Test ROC",       os.path.join(opt_dir, "roc_test.png"))

    pd.DataFrame([{
        "best_lr":best_lr, "best_batch":best_bs,
        "train_auc":tr_auc, "val_auc":va_auc, "test_auc":te_auc,
        **{f"val_{k}":v for k,v in va_stats.items()},
        **{f"test_{k}":v for k,v in te_stats.items()}
    }]).to_csv(os.path.join(opt_dir, "metrics.csv"), index=False)

    print("\n✓ Finished.  Results saved under:", results_root)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main(parse_args())
