#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
score_original.py – compute connectivity‐corrected scores for the
original model using your existing pipeline exactly as before.
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from openbinn.binn.data import PnetSimDataSet, ReactomeNetwork, get_layer_maps

def connectivity_corrected_scores(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for layer, sub in df.groupby('layer', sort=False):
        imp = sub['importance']
        deg = sub['degree']

        imp_std = imp.std(ddof=0)
        deg_std = deg.std(ddof=0)

        ZA = (imp - imp.mean()) / (imp_std if imp_std != 0 else 1.0)
        Zd = (deg - deg.mean()) / (deg_std if deg_std != 0 else 1.0)

        S = ZA / Zd.replace(0, np.nan)

        temp = sub.copy()
        temp['Z_importance'] = ZA
        temp['Z_degree']     = Zd
        temp['score']        = S

        results.append(temp)

    return pd.concat(results, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   help="Original model folder (must contain explanations/)")
    p.add_argument("--method", default="deeplift",
                   choices=["deeplift","ig","lime","shap"])
    args = p.parse_args()

    base = Path(args.scenario)
    data_name = base.name  # e.g. "original"
    importance_path = base / "explanations"
    out_csv = base / f"{data_name}_{args.method}_target_scores.csv"

    # 1) aggregate layer importance exactly as before
    layer_importance_list = []
    for layer in range(0, 7):
        importance_list = []
        for i in range(1, 8):
            if layer < i:
                fn = importance_path / f"PNet_{data_name}_{args.method}_target_{i}_layer_{layer}_test.csv"
                if not fn.exists():
                    continue
                df = pd.read_csv(fn, index_col=-1)
                importance_list.append(df)
        if not importance_list:
            continue

        agg = np.abs(importance_list[0].copy())
        for imp_df in importance_list[1:]:
            agg.iloc[:, :-2] += np.abs(imp_df.iloc[:, :-2])

        layer_imp = pd.DataFrame(agg.iloc[:, :-2].sum(axis=0), columns=['importance'])
        layer_imp['layer'] = layer
        layer_importance_list.append(layer_imp)

    layer_importance_df = pd.concat(layer_importance_list, axis=0)

    # 2) build maps once (for degree calculation)
    ds = PnetSimDataSet(root=str(base), num_features=1)
    reactome = ReactomeNetwork(dict(
        reactome_base_dir="./biological_knowledge/reactome",
        relations_file_name="ReactomePathwaysRelation.txt",
        pathway_names_file_name="ReactomePathways.txt",
        pathway_genes_file_name="ReactomePathways.gmt",
    ))
    maps = get_layer_maps(
        genes=list(ds.node_index), reactome=reactome,
        n_levels=6, direction="root_to_leaf", add_unk_genes=False
    )

    # 3) compute in/out degrees
    genes = layer_importance_df[layer_importance_df['layer'] == 0].index.tolist()
    in_degrees, out_degrees = [], []

    # layer 0: genes
    in_degrees.append(pd.DataFrame(0, index=genes, columns=["in_degree"]))
    out_degrees.append(pd.DataFrame(
        maps[0].sum(axis=1).astype(float), columns=["out_degree"]))

    # layers 1..6
    for i in range(1, len(maps)):
        parent_df = maps[i-1]
        child_df  = maps[i]

        in_degrees.append(pd.DataFrame(
            parent_df.sum(axis=0).astype(float), columns=["in_degree"]))
        out_degrees.append(pd.DataFrame(
            child_df.sum(axis=1).astype(float), columns=["out_degree"]))

    assert len(in_degrees) == len(out_degrees) == 7

    in_degrees_df  = pd.concat(in_degrees, axis=0)
    out_degrees_df = pd.concat(out_degrees, axis=0)

    degree_df = pd.concat([in_degrees_df, out_degrees_df], axis=1)
    degree_df['degree'] = degree_df['in_degree'] + degree_df['out_degree']

    # 4) merge & score
    importance_degree_df = pd.concat([layer_importance_df, degree_df], axis=1)
    df_scored = connectivity_corrected_scores(importance_degree_df)

    # 5) save
    df_scored.to_csv(out_csv)
    print(f"Saved connectivity‐corrected scores to {out_csv}")


if __name__ == "__main__":
    main()
