#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_explanation.py – compute explanations on **all** layers
of bootstrap, gene-perm, and label-perm PNet models on the test split.

이전 로직을 그대로 따라가되, wrapper 정의 위치와 컬럼 매핑을 보완했습니다.
"""
import os, time, argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader

from openbinn.binn import PNet
from openbinn.binn.data import PnetSimExpDataSet, ReactomeNetwork, get_layer_maps
from openbinn.explainer import Explainer
import openbinn.experiment_utils as utils


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: PNet, target_layer: int):
        super().__init__()
        self.model = model
        # Explainer가 이 속성을 참조합니다
        self.print_layer = target_layer
        self.target_layer = target_layer

    def forward(self, x):
        outs = self.model(x)
        return outs[self.target_layer - 1]


def explain_one_variant(seed_dir: Path, method: str):
    print(f"→ Explaining {seed_dir}")
    # 1) dataset & splits
    ds = PnetSimExpDataSet(root=str(seed_dir), num_features=1)
    split_fp = seed_dir / "splits"
    ds.split_index_by_file(
        train_fp=str(split_fp / "training_set_0.csv"),
        valid_fp=str(split_fp / "validation_set.csv"),
        test_fp =str(split_fp / "test_set.csv"),
    )

    # 2) Reactome maps
    reactome = ReactomeNetwork({
        "reactome_base_dir":"./biological_knowledge/reactome",
        "relations_file_name":"ReactomePathwaysRelation.txt",
        "pathway_names_file_name":"ReactomePathways.txt",
        "pathway_genes_file_name":"ReactomePathways.gmt",
    })
    maps = get_layer_maps(
        genes=list(ds.node_index), reactome=reactome,
        n_levels=6, direction="root_to_leaf", add_unk_genes=False
    )

    # 3) load trained model
    model_fp = seed_dir / "results" / "trained_model.pth"
    model = PNet(layers=maps, num_genes=maps[0].shape[0], lr=0.001)
    state = torch.load(model_fp, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 4) test loader
    torch.manual_seed(42)
    loader = DataLoader(
        ds,
        batch_size=len(ds.test_idx),
        sampler=SubsetRandomSampler(ds.test_idx),
        num_workers=0
    )

    # 5) explainer config
    config = utils.load_config("experiment_config.json")
    folder_name = seed_dir / "explanations"
    folder_name.mkdir(exist_ok=True)

    model_name = "PNet"
    data_name  = seed_dir.parent.name  # "bootstrap" 등
    split_name = "test"

    # 6) for each target_layer
    for target_layer in range(1, len(maps) + 1):
        print(f"  → Layer {target_layer}")
        wrapped_model = ModelWrapper(model, target_layer)

        explanations_accum = {}
        labels_accum       = []
        predictions_accum  = []
        sample_ids_accum   = []

        for batch in loader:
            inputs, labels, sample_ids = batch
            inputs = inputs.float()
            labels = labels.long()

            # explainer params
            param_dict = utils.fill_param_dict(method, config['explainers'][method], inputs)
            baseline = torch.zeros_like(inputs)
            param_dict['baseline'] = baseline 
            param_dict['classification_type'] = 'binary'

            explainer = Explainer(method, wrapped_model, param_dict)
            explanation_dict = explainer.get_layer_explanations(inputs, labels)

            # accumulate per layer_name
            for layer_name, tensor in explanation_dict.items():
                explanations_accum.setdefault(layer_name, []).append(tensor.detach().cpu().numpy())

            labels_accum.append(labels.detach().cpu().numpy())
            predictions_accum.append(wrapped_model(inputs).detach().cpu().numpy())
            sample_ids_accum.append(sample_ids)

        # merge & save for this target_layer
        for layer_index, (layer_name, list_of_arrays) in enumerate(explanations_accum.items()):
            if layer_index >= len(maps):
                break

            explanation_array = np.concatenate(list_of_arrays, axis=0)
            all_labels       = np.concatenate(labels_accum, axis=0)
            all_predictions  = np.concatenate(predictions_accum, axis=0)
            all_sample_ids   = [sid for batch in sample_ids_accum for sid in batch]

            # choose correct axis from map
            cur_map = maps[layer_index]
            W = explanation_array.shape[1]
            if cur_map.shape[0] == W:
                column_names = list(cur_map.index)
            elif cur_map.shape[1] == W:
                column_names = list(cur_map.columns)
            else:
                raise ValueError(
                    f"Layer map shape {cur_map.shape} != expl width {W}"
                )

            df = pd.DataFrame(explanation_array, columns=column_names)
            df['label']      = all_labels
            df['prediction'] = all_predictions
            df['sample_id']  = all_sample_ids

            file_path = folder_name / f"{model_name}_{data_name}_{method}_target_{target_layer}_layer_{layer_index}_{split_name}.csv"
            df.to_csv(file_path, index=False)
            print(f"    saved {file_path}")

    print(f"Finished explaining {seed_dir}\n")


def main(args):
    scenario = Path(args.scenario)
    for variant in ("bootstrap", "gene-perm"):
        root = scenario / variant
        if not root.exists():
            continue
        print(f"\n=== Variant: {variant} ===")
        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name)
            except ValueError:
                continue
            if not (args.seed_start <= seed <= args.seed_end):
                continue
            explain_one_variant(seed_dir, method=args.method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="Base scenario dir")
    parser.add_argument("--method",   default="deeplift",
                        choices=["deeplift", "ig", "lime", "shap"])
    parser.add_argument("--seed_start", type=int, default=100, help="First seed inclusive")
    parser.add_argument("--seed_end",   type=int, default=200, help="Last seed inclusive")
    args = parser.parse_args()
    main(args)
