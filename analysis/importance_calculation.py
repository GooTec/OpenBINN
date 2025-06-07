#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
explain_all_variants.py
───────────────────────
◆ 입력  : data/b4_g4/{i}/[..., bootstrap/{b}, gene-permutation/{b}, label-permutation/{b}]
          └─ results/optimal/trained_model.pth      (〈train_all_variants_fast.py〉 산출물)
◆ 출력  : explanations/*.csv  (레이어별 원시 explain) +
          PNet_{method}_target_scores.csv (집계·표준화 중요도)
"""

from pathlib import Path
import os, warnings, argparse, time

import torch
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader

from openxai.binn import PNet
from openxai.binn.data import PnetSimExpDataSet, ReactomeNetwork, get_layer_maps
from openxai.explainer import Explainer
import openxai.experiment_utils as utils

# ──────────────────────────────────────────
METHOD        = "deeplift"   # ← ig, lime, shap 등으로 변경 가능
N_SIM         = 100
N_VARIANTS    = 100
DEFAULT_BETA  = 2
DEFAULT_GAMMA = 2
DATA_ROOT     = Path(f"./data/b{DEFAULT_BETA}_g{DEFAULT_GAMMA}")
NUM_WORKERS   = 0
SEED          = 42
# ──────────────────────────────────────────


# ╭─────────────────────────── 헬퍼들 ───────────────────────────╮
class ModelWrapper(torch.nn.Module):
    def __init__(self, model: PNet, target_layer: int):
        super().__init__()
        self.model        = model
        self.print_layer  = target_layer
        self.target_layer = target_layer
    def forward(self, x):
        outs = self.model(x)
        return outs[self.target_layer - 1]


def connectivity_corrected_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, sub in df.groupby('layer', sort=False):
        imp = sub['importance']
        std = imp.std(ddof=0)
        ZA  = (imp - imp.mean()) / (std if std != 0 else 1.0)
        tmp = sub.copy(); tmp['Z_importance'] = ZA
        rows.append(tmp)
    return pd.concat(rows, axis=0)


def load_reactome_once():
    return ReactomeNetwork(dict(
        reactome_base_dir="../biological_knowledge/simulation",
        relations_file_name="SimulationPathwaysRelation.txt",
        pathway_names_file_name="SimulationPathways.txt",
        pathway_genes_file_name="SimulationPathways.gmt",
    ))
# ╰─────────────────────────────────────────────────────────────╯


def explain_dataset(scen_dir: Path, reactome):
    """단일 폴더 (original/variant) 에 대한 모든 중요도 계산."""
    model_fp = scen_dir / "results" / "optimal" / "trained_model.pth"
    split_dir = scen_dir / "splits"
    if not model_fp.exists() or not split_dir.exists():
        print(f"[skip] model 또는 splits 없음 → {scen_dir.relative_to(DATA_ROOT)}")
        return

    # ─ 데이터·맵 ──────────────────────
    ds = PnetSimExpDataSet(root=str(scen_dir), num_features=1)
    ds.split_index_by_file(
        train_fp=split_dir/"training_set_0.csv",
        valid_fp=split_dir/"validation_set.csv",
        test_fp =split_dir/"test_set.csv",
    )

    maps = get_layer_maps(
        genes=list(ds.node_index), reactome=reactome,
        n_levels=3, direction="root_to_leaf", add_unk_genes=False
    )
    ds.node_index = [g for g in ds.node_index if g in maps[0].index]

    # ─ 모델 ───────────────────────────
    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=0.001)
    state = torch.load(model_fp, map_location="cpu")
    model.load_state_dict(state); model.eval()

    # ─ DataLoader (전체 test 한번에) ───
    torch.manual_seed(SEED)
    test_loader = DataLoader(
        ds, batch_size=len(ds.test_idx),
        sampler=SubsetRandomSampler(ds.test_idx),
        num_workers=NUM_WORKERS
    )

    config       = utils.load_config("../experiment_config.json")
    explain_root = scen_dir / "explanations"; explain_root.mkdir(exist_ok=True)
    model_name   = "PNet"
    data_label   = scen_dir.name        # ex) 1, 37, 5 …
    split_name   = "test"

    # ───────────────── layer loop ──────────────────
    for tgt in range(1, len(maps)+1):
        wrap = ModelWrapper(model, tgt)

        expl_acc, lab_acc, pred_acc, id_acc = {}, [], [], []
        for X, y, ids in test_loader:
            X = X.float(); y = y.long()
            p_conf = utils.fill_param_dict(METHOD, config['explainers'][METHOD], X)
            p_conf['baseline']             = torch.zeros_like(X)
            p_conf['classification_type']  = 'binary'

            explainer = Explainer(METHOD, wrap, p_conf)
            exp_dict  = explainer.get_layer_explanations(X, y)

            for lname, ten in exp_dict.items():
                expl_acc.setdefault(lname, []).append(ten.detach().cpu().numpy())
            lab_acc.append(y.cpu().numpy())
            pred_acc.append(wrap(X).detach().cpu().numpy())
            id_acc.append(ids)

        # ─ save per-layer CSV ───────────
        for idx, (lname, arrs) in enumerate(expl_acc.items()):
            if idx >= len(maps): break
            expl_arr = np.concatenate(arrs, axis=0)
            labels   = np.concatenate(lab_acc,  axis=0)
            preds    = np.concatenate(pred_acc, axis=0)
            all_ids  = [sid for batch in id_acc for sid in batch]

            cur_map = maps[idx]; W = expl_arr.shape[1]
            cols = list(cur_map.index) if cur_map.shape[0]==W else list(cur_map.columns)

            df = pd.DataFrame(expl_arr, columns=cols)
            df['label']      = labels
            df['prediction'] = preds
            df['sample_id']  = all_ids

            csv_fp = explain_root / f"{model_name}_{data_label}_{METHOD}_L{tgt}_layer{idx}_{split_name}.csv"
            df.to_csv(csv_fp, index=False)

    # ───────────────── 중요도 집계 & 스코어 ─────────────────
    layer_imp_list = []
    for layer in range(4):                         # layer 0‒3
        imps = []
        for tgt in range(1,5):
            fp = explain_root / f"{model_name}_{data_label}_{METHOD}_L{tgt}_layer{layer}_{split_name}.csv"
            if fp.exists():
                imps.append(pd.read_csv(fp, index_col=-1))
        if not imps: continue
        agg = np.abs(imps[0].iloc[:, :-2]).copy()
        for df in imps[1:]:
            agg += np.abs(df.iloc[:, :-2])
        layer_imp_list.append(pd.DataFrame({
            'importance': agg.sum(axis=0), 'layer': layer
        }))
    layer_imp_df = pd.concat(layer_imp_list, axis=0)
    scored_df    = connectivity_corrected_scores(layer_imp_df)

    out_fp = scen_dir / f"PNet_{METHOD}_target_scores.csv"
    scored_df.to_csv(out_fp, index=True)
    print(f"    ✓ scores saved → {out_fp.relative_to(DATA_ROOT)}")


# ╭────────────────────────────────────────────────────────────╮
# │                           main                            │
# ╰────────────────────────────────────────────────────────────╯
def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    ap = argparse.ArgumentParser()
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim",   type=int, default=N_SIM)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument(
        "--statistical_method",
        choices=["bootstrap", "gene-permutation", "label-permutation", "all"],
        default="all",
        help="Variant type to process when computing importances.",
    )
    ap.add_argument(
        "--skip_original",
        action="store_true",
        help="Do not calculate importances for original datasets",
    )
    args = ap.parse_args()

    data_root = Path(f"./data/b{args.beta}_g{args.gamma}")

    reactome = load_reactome_once()

    if args.statistical_method == "all":
        variants = ["bootstrap", "gene-permutation", "label-permutation"]
    else:
        variants = [args.statistical_method]

    for i in range(args.start_sim, args.end_sim + 1):
        base = data_root / f"{i}"
        print(f"\n■■ Simulation {i:3d} ■■")
        if not args.skip_original:
            explain_dataset(base, reactome)  # original

        for vtype in variants:
            for b in range(1, N_VARIANTS + 1):
                vdir = base / vtype / f"{b}"
                if vdir.exists():
                    print(f"  → {vtype}/{b}")
                    explain_dataset(vdir, reactome)

    print("\n✓ 모든 중요도 계산 완료.")

if __name__ == "__main__":
    main()
