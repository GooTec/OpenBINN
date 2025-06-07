#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fit_variants_pnets.py – fit OPENXAI-PNet on bootstrap, gene-perm, and label-perm datasets

For each variant type (`bootstrap`, `gene-perm`, `label-perm`) and each seed under
    {scenario}/{variant}/{seed}/
this script will:

1. Symlink all metadata (`gene_alpha.csv`, `pathway_beta.csv`, `pca_plot.png`, `seeds.txt`,
   `selected_genes.csv`, and the `splits/` dir) from the base scenario into the seed folder
2. Instantiate PnetSimDataSet on that folder
3. Train PNet with lr=0.001, batch_size=8, max_epochs=100 (no progress bar)
4. Evaluate train & val AUROC
5. Save under {scenario}/{variant}/{seed}/results/:
     - trained_model.pth
     - roc_train.png
     - roc_val.png
     - metrics.csv
"""
import argparse, os
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as GeoLoader
from openbinn.binn import PNet
from openbinn.binn.util import InMemoryLogger, get_roc
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps


def ensure_symlinks(seed_dir: Path, base_dir: Path):
    """Symlink metadata files from base_dir into seed_dir (excluding mutation/response)."""
    items = [
        "gene_alpha.csv",
        "pathway_beta.csv",
        "pca_plot.png",
        "seeds.txt",
        "selected_genes.csv",
        "splits",
    ]
    for name in items:
        src = base_dir / name
        dst = seed_dir / name
        if dst.exists():
            continue
        rel = os.path.relpath(src, seed_dir)
        dst.symlink_to(rel, target_is_directory=src.is_dir())


def fit_one_variant(seed_dir: Path, lr: float, batch_size: int, max_epochs: int):
    base_dir = seed_dir.parents[1]
    ensure_symlinks(seed_dir, base_dir)

    # build dataset
    ds = PnetSimDataSet(root=str(seed_dir), num_features=1)
    splits = seed_dir / "splits"
    ds.split_index_by_file(
        train_fp=str(splits / "training_set_0.csv"),
        valid_fp=str(splits / "validation_set.csv"),
        test_fp =str(splits / "test_set.csv"),
    )

    # hierarchy maps
    reactome = ReactomeNetwork(dict(
        reactome_base_dir="./biological_knowledge/reactome",
        relations_file_name="ReactomePathwaysRelation.txt",
        pathway_names_file_name="ReactomePathways.txt",
        pathway_genes_file_name="ReactomePathways.gmt",
    ))
    maps = get_layer_maps(
        genes=list(ds.node_index), reactome=reactome,
        n_levels=6, direction="root_to_leaf", add_unk_genes=False)

    # dataloaders
    train_loader = GeoLoader(ds, batch_size,
                             sampler=torch.utils.data.SubsetRandomSampler(ds.train_idx))
    val_loader   = GeoLoader(ds, batch_size,
                             sampler=torch.utils.data.SubsetRandomSampler(ds.valid_idx))

    # trainer
    pl.seed_everything(42, workers=True)
    model   = PNet(layers=maps, num_genes=maps[0].shape[0], lr=lr)
    trainer = pl.Trainer(
        accelerator="auto", deterministic=True,
        max_epochs=max_epochs,
        logger=InMemoryLogger(),
        enable_progress_bar=False
    )
    trainer.fit(model, train_loader, val_loader)

    # compute ROC
    fpr_tr, tpr_tr, tr_auc, _, _ = get_roc(model, train_loader, exp=False)
    fpr_v,  tpr_v,  val_auc,  _, _ = get_roc(model, val_loader,   exp=False)

    # save results
    res_dir = seed_dir / "results"
    res_dir.mkdir(exist_ok=True)

    # model
    torch.save(model.state_dict(), res_dir / "trained_model.pth")

    # ROC curves
    import matplotlib.pyplot as plt
    def save_roc(fpr, tpr, auc, title, path):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUROC={auc:.3f}")
        ax.plot([0,1],[0,1],"--",c="grey")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(path); plt.close(fig)

    save_roc(fpr_tr, tpr_tr, tr_auc, "Train ROC",      res_dir / "roc_train.png")
    save_roc(fpr_v,  tpr_v,  val_auc, "Validation ROC", res_dir / "roc_val.png")

    # metrics
    import pandas as pd
    pd.DataFrame([{"train_auc": tr_auc, "val_auc": val_auc}])\
        .to_csv(res_dir / "metrics.csv", index=False)


def main(args):
    scenario = Path(args.scenario)
    for variant in ("gene-perm", "label-perm"):
        root = scenario / variant
        if not root.exists():
            continue
        print(f"\n=== Variant: {variant} ===")
        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir():
                continue
            print(f"  Seed {seed_dir.name} → fitting…")
            fit_one_variant(seed_dir, lr=0.001, batch_size=8, max_epochs=100)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   help="Base scenario directory containing bootstrap/, gene-perm/, label-perm/")
    args = p.parse_args()
    main(args)
