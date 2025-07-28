#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_all_variants_fast.py  (CSVLogger 버전)
────────────────────────────────────────────
• b4_g4 시뮬레이션(1‒100)  →  원본 데이터에 대해 한 번 Grid-Search
  → best (lr*, bs*) 도출  → 그 값으로 변형 300 세트 학습.
  · 로그/체크포인트는 기본적으로 ./pl_runs/… 이하에 저장
    (폴더가 이미 있으면 _1, _2 … 번호를 붙여 새로 생성)
  · --log_dir 옵션으로 루트 폴더를 변경 가능
  · 모든 PyTorch Lightning 로그는 CSVLogger 로 기록
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
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score
)

from openbinn.binn import PNet
from openbinn.binn.util import get_roc, eval_metrics
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

# ───────────────────────────────────
LR_LIST     = [1e-3, 5e-3]
BS_LIST     = [8, 16]
MAX_EPOCHS  = 200
PATIENCE    = 10
N_SIM       = 100
N_VARIANTS  = 100
DATA_ROOT   = Path("./data/b0_g0.0")
NUM_WORKERS = 0
# ───────────────────────────────────


# ╭──────────────────────────────────────────────────────────────╮
# │               중복되지 않는 로그 폴더 경로 생성               │
# ╰──────────────────────────────────────────────────────────────╯
def get_unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for idx in range(1, 10_000):
        cand = Path(f"{base}_{idx}")
        if not cand.exists():
            return cand
    raise RuntimeError("더 이상의 고유 폴더 이름을 생성할 수 없습니다.")


# ╭──────────────────────────────────────────────────────────────╮
# │                       보조 함수들                             │
# ╰──────────────────────────────────────────────────────────────╯
def bin_stats(y, p):
    pred = p[:, 1] > 0.5
    return dict(
        acc=accuracy_score(y, pred),
        aupr=average_precision_score(y, p[:, 1]),
        f1=f1_score(y, pred),
        prec=precision_score(y, pred),
        rec=recall_score(y, pred),
    )


def save_roc(fpr, tpr, auc_val, title, path):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUROC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def load_reactome_once():
    return ReactomeNetwork(
        dict(
            reactome_base_dir="../biological_knowledge/simulation",
            relations_file_name="SimulationPathwaysRelation.txt",
            pathway_names_file_name="SimulationPathways.txt",
            pathway_genes_file_name="SimulationPathways.gmt",
        )
    )


# ╭──────────────────────────────────────────────────────────────╮
# │  train_dataset():                                            │
# │  · best_params = None  → Grid search + 최적 학습 + 반환      │
# │  · best_params = (lr*, bs*) → 그 값으로만 학습·평가          │
# ╰──────────────────────────────────────────────────────────────╯
def train_dataset(
    scen_dir: Path,
    reactome,
    log_root: Path,
    best_params=None,
):
    if not (scen_dir / "splits").exists():
        print(f"[WARN] splits 없음: {scen_dir}")
        return best_params

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

    log_dir = (
        log_root
        / f"sim_{scen_dir.relative_to(DATA_ROOT).as_posix().replace('/', '_')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    results_root = scen_dir / "results"
    opt_dir = results_root / "optimal"
    opt_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────
    # ① Grid-Search (original 에서만)
    # ────────────────────────────────
    if best_params is None:
        summary_rows = []
        for lr, bs in product(LR_LIST, BS_LIST):
            tag = f"lr_{lr:g}_bs_{bs}"
            print(f"      Grid ▶ {scen_dir.relative_to(DATA_ROOT)} | {tag}")

            tr_loader = GeoLoader(
                ds, bs, sampler=SubsetRandomSampler(ds.train_idx), num_workers=NUM_WORKERS
            )
            va_loader = GeoLoader(
                ds, bs, sampler=SubsetRandomSampler(ds.valid_idx), num_workers=NUM_WORKERS
            )

            model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=lr,
                        diversity_lambda=0.1)

            logger = CSVLogger(save_dir=str(log_dir / "grid"), name=tag)
            trainer = pl.Trainer(
                accelerator="auto",
                deterministic=True,
                max_epochs=MAX_EPOCHS,
                callbacks=[
                    EarlyStopping(
                        "val_loss", patience=PATIENCE, mode="min", verbose=False, min_delta=0.01
                    )
                ],
                logger=logger,
                enable_progress_bar=False,
                default_root_dir=str(log_dir),
            )

            t0 = time.time()
            trainer.fit(model, tr_loader, va_loader)
            run_t = time.time() - t0

            _, _, tr_auc, _, _ = get_roc(model, tr_loader, exp=False)
            fpr, tpr, va_auc, yv, pv = get_roc(model, va_loader, exp=False)

            exp_dir = results_root / "hparam_search" / tag
            exp_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), exp_dir / "trained_model.pth")
            save_roc(fpr, tpr, va_auc, "Validation ROC", exp_dir / "roc_valid.png")

            summary_rows.append(
                {
                    "lr": lr,
                    "bs": bs,
                    "val_auc": va_auc,
                    "runtime": run_t,
                    "train_auc": tr_auc,
                    **{f"val_{k}": v for k, v in bin_stats(yv, pv).items()},
                }
            )

        df = pd.DataFrame(summary_rows).sort_values("val_auc", ascending=False)
        best_lr, best_bs = df.iloc[0][["lr", "bs"]].tolist()
        best_params = (best_lr, int(best_bs))

    # ────────────────────────────────
    # ② 고정 파라미터로 학습 & 평가
    # ────────────────────────────────
    best_lr, best_bs = best_params

    tr_loader = GeoLoader(
        ds, best_bs, sampler=SubsetRandomSampler(ds.train_idx), num_workers=NUM_WORKERS
    )
    va_loader = GeoLoader(
        ds, best_bs, sampler=SubsetRandomSampler(ds.valid_idx), num_workers=NUM_WORKERS
    )
    te_loader = GeoLoader(
        ds, best_bs, sampler=SubsetRandomSampler(ds.test_idx), num_workers=NUM_WORKERS
    )

    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=best_lr,
                 diversity_lambda=0.1)

    logger = CSVLogger(save_dir=str(log_dir / "optimal"), name="")

    init_loss, init_acc, init_auc = eval_metrics(model, va_loader)
    print(f"      Start: loss={init_loss:.4f} acc={init_acc:.4f} auc={init_auc:.4f}")
    trainer = pl.Trainer(
        accelerator="auto",
        deterministic=True,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            EarlyStopping(
                "val_loss", patience=PATIENCE, mode="min", verbose=False, min_delta=0.01
            )
        ],
        logger=logger,
        enable_progress_bar=False,
        default_root_dir=str(log_dir),
    )
    trainer.fit(model, tr_loader, va_loader)
    fin_loss, fin_acc, fin_auc = eval_metrics(model, va_loader)
    print(f"      End  : loss={fin_loss:.4f} acc={fin_acc:.4f} auc={fin_auc:.4f}")

    _, _, tr_auc, _, _ = get_roc(model, tr_loader, exp=False)
    fv, tv, va_auc, yv, pv = get_roc(model, va_loader, exp=False)
    ft, tt, te_auc, yt, pt = get_roc(model, te_loader, exp=False)

    torch.save(model.state_dict(), opt_dir / "trained_model.pth")
    save_roc(fv, tv, va_auc, "Validation ROC", opt_dir / "roc_valid.png")
    save_roc(ft, tt, te_auc, "Test ROC", opt_dir / "roc_test.png")

    pd.DataFrame(
        [
            {
                "best_lr": best_lr,
                "best_bs": best_bs,
                "train_auc": tr_auc,
                "val_auc": va_auc,
                "test_auc": te_auc,
                **{f"val_{k}": v for k, v in bin_stats(yv, pv).items()},
                **{f"test_{k}": v for k, v in bin_stats(yt, pt).items()},
            }
        ]
    ).to_csv(opt_dir / "metrics.csv", index=False)

    return best_params  # (lr*, bs*)


# ╭──────────────────────────────────────────────────────────────╮
# │                           main()                             │
# ╰──────────────────────────────────────────────────────────────╯
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    pl.seed_everything(42, workers=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim", type=int, default=N_SIM)
    ap.add_argument(
        "--log_dir",
        type=str,
        default="./pl_runs",
        help="로그를 저장할 루트 폴더 (없으면 생성, 이미 있으면 _1 등 번호를 붙여 새로 생성)",
    )
    args = ap.parse_args()

    log_root_base = Path(args.log_dir).expanduser().resolve()
    log_root = get_unique_dir(log_root_base)
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 로그 저장 위치: {log_root}")

    reactome = load_reactome_once()

    for i in range(args.start_sim, args.end_sim + 1):
        base_dir = DATA_ROOT / f"{i}"
        print(f"\n■■ Simulation {i:3d} ■■")

        # ① original → grid search
        best_params = train_dataset(
            base_dir, reactome, log_root=log_root, best_params=None
        )

        # ② variants → 고정 best_params
        for vtype in ["gene-permutation"]:
            for b in range(1, N_VARIANTS + 1):
                v_dir = base_dir / vtype / f"{b}"
                if v_dir.exists():
                    print(f"    Variant ▶ {vtype}/{b}")
                    train_dataset(
                        v_dir, reactome, log_root=log_root, best_params=best_params
                    )

    print("\n✓ 모든 데이터셋 학습 완료.")


if __name__ == "__main__":
    main()
