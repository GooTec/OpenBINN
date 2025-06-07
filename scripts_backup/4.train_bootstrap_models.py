#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fit_bootstrapped_pnets.py – fit OPENXAI-PNet on bootstrap variants with fixed hyperparams

Each bootstrap/{seed}/ 폴더 안에 base 디렉터리의 주요 파일을
심볼릭 링크로 연결하되, mutation/response 파일은 링크하지 않습니다.
"""
import argparse, os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as GeoLoader
import pytorch_lightning as pl
from openbinn.binn import PNet
from openbinn.binn.util import ProgressBar, InMemoryLogger, get_roc
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

def ensure_symlinks(bs_dir: Path, base_dir: Path):
    # 링크할 대상들 (mutation/response 제외)
    items = [
        "gene_alpha.csv",
        "pathway_beta.csv",
        "pca_plot.png",
        "seeds.txt",
        "selected_genes.csv",
        "splits",  # 디렉터리 통째로
    ]
    for name in items:
        src = base_dir / name
        dst = bs_dir / name
        if dst.exists():
            continue
        rel = os.path.relpath(src, bs_dir)
        dst.symlink_to(rel, target_is_directory=src.is_dir())

def fit_one_bootstrap(bs_dir: Path, lr: float, batch_size: int, max_epochs: int):
    base_dir = bs_dir.parents[1]
    ensure_symlinks(bs_dir, base_dir)

    # PNet 데이터셋 생성
    ds = PnetSimDataSet(root=str(bs_dir), num_features=1)
    splits = bs_dir / "splits"
    ds.split_index_by_file(
        train_fp=str(splits / "training_set_0.csv"),
        valid_fp=str(splits / "validation_set.csv"),
        test_fp =str(splits / "test_set.csv"),
    )

    # Reactome hierarchy 매핑
    reactome = ReactomeNetwork(dict(
        reactome_base_dir="./biological_knowledge/reactome",
        relations_file_name="ReactomePathwaysRelation.txt",
        pathway_names_file_name="ReactomePathways.txt",
        pathway_genes_file_name="ReactomePathways.gmt",
    ))
    maps = get_layer_maps(
        genes=list(ds.node_index), reactome=reactome,
        n_levels=6, direction="root_to_leaf", add_unk_genes=False)

    # DataLoader
    train_loader = GeoLoader(ds, batch_size,
                             sampler=torch.utils.data.SubsetRandomSampler(ds.train_idx))
    val_loader   = GeoLoader(ds, batch_size,
                             sampler=torch.utils.data.SubsetRandomSampler(ds.valid_idx))

    # Trainer 설정
    pl.seed_everything(42, workers=True)
    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=lr)
    trainer = pl.Trainer(
        accelerator="auto", deterministic=True,
        max_epochs=max_epochs,
        # ProgressBar 콜백 제거
        logger=InMemoryLogger(), 
        enable_progress_bar=False
    )

    trainer.fit(model, train_loader, val_loader)

    # ROC 계산
    fpr_tr, tpr_tr, tr_auc, _, _ = get_roc(model, train_loader, exp=False)
    fpr_v,  tpr_v,  val_auc,  _, _ = get_roc(model, val_loader,   exp=False)

    # 결과 저장
    res_dir = bs_dir / "results"
    res_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), res_dir / "trained_model.pth")

    import matplotlib.pyplot as plt
    def save_roc(fpr, tpr, auc, title, path):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUROC={auc:.3f}")
        ax.plot([0,1], [0,1], "--", c="grey")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    save_roc(fpr_tr, tpr_tr, tr_auc, "Train ROC",      res_dir / "roc_train.png")
    save_roc(fpr_v,  tpr_v,  val_auc, "Validation ROC", res_dir / "roc_val.png")

    import pandas as pd
    pd.DataFrame([{"train_auc": tr_auc, "val_auc": val_auc}])\
      .to_csv(res_dir / "metrics.csv", index=False)

def main(args):
    scenario = Path(args.scenario)
    bs_root = scenario / "bootstrap"
    for seed_dir in sorted(bs_root.iterdir()):
        if not seed_dir.is_dir(): continue
        print(f"Processing bootstrap seed {seed_dir.name}…")
        fit_one_bootstrap(seed_dir, lr=0.001, batch_size=8, max_epochs=100)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   help="Base scenario folder containing bootstrap/")
    args = p.parse_args()
    main(args)
