#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
score_bootstrap.py – compute connectivity‐corrected scores for each
bootstrap/{seed} run, using your existing pipeline exactly as before,
단, degree_df 생성 로직만 수정했습니다.
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
        S  = ZA / Zd.replace(0, np.nan)

        temp = sub.copy()
        temp['Z_importance'] = ZA
        temp['Z_degree']     = Zd
        temp['score']        = S
        results.append(temp)

    return pd.concat(results, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",   required=True,
                   help="Base scenario folder, e.g. simulation/att10/linear")
    p.add_argument("--method",     default="deeplift",
                   choices=["deeplift","ig","lime","shap"])
    p.add_argument("--seed_start", type=int, default=100,
                   help="First bootstrap seed inclusive")
    p.add_argument("--seed_end",   type=int, default=200,
                   help="Last bootstrap seed inclusive")
    args = p.parse_args()

    base    = Path(args.scenario)
    variant = "bootstrap"
    method  = args.method

    for seed in range(args.seed_start, args.seed_end + 1):
        seed_dir = base / variant / str(seed)
        if not seed_dir.is_dir():
            continue

        print(f"--- Seed {seed} ---")
        imp_dir = seed_dir / "explanations"
        out_csv = seed_dir / "scores.csv"

        # 1) aggregate layer importance
        layer_importance_list = []
        for layer in range(0, 7):
            importance_list = []
            for i in range(1, 8):
                if layer < i:
                    fn = imp_dir / f"PNet_{variant}_{method}_target_{i}_layer_{layer}_test.csv"
                    if not fn.exists():
                        continue
                    df = pd.read_csv(fn, index_col=-1)
                    importance_list.append(df)
            if not importance_list:
                continue

            agg = np.abs(importance_list[0]).iloc[:, :-2].copy()
            for imp_df in importance_list[1:]:
                agg += np.abs(imp_df.iloc[:, :-2])

            layer_imp = pd.DataFrame(agg.sum(axis=0), columns=['importance'])
            layer_imp['layer'] = layer
            layer_importance_list.append(layer_imp)

        if not layer_importance_list:
            print(f"Seed {seed}: no attributions found, skipping.")
            continue

        layer_importance_df = pd.concat(layer_importance_list, axis=0)

        # 2) build maps once per seed
        ds = PnetSimDataSet(root=str(seed_dir), num_features=1)
        reactome = ReactomeNetwork({
            'reactome_base_dir': './biological_knowledge/reactome',
            'relations_file_name': 'ReactomePathwaysRelation.txt',
            'pathway_names_file_name': 'ReactomePathways.txt',
            'pathway_genes_file_name': 'ReactomePathways.gmt',
        })
        maps = get_layer_maps(
            genes=list(ds.node_index), reactome=reactome,
            n_levels=6, direction='root_to_leaf', add_unk_genes=False
        )

        # 3) compute in/out degrees per layer and join them
        in_degrees, out_degrees = [], []
        # layer 0: genes
        genes = layer_importance_df[layer_importance_df.layer == 0].index
        in_degrees.append(pd.DataFrame(0, index=genes, columns=['in_degree']))
        out_degrees.append(pd.DataFrame(
            maps[0].sum(axis=1).astype(float), columns=['out_degree']))

        # layers 1..6
        for i in range(1, len(maps)):
            parent_df = maps[i-1]
            child_df  = maps[i]

            in_degrees.append(pd.DataFrame(
                parent_df.sum(axis=0).astype(float), columns=['in_degree']))
            out_degrees.append(pd.DataFrame(
                child_df.sum(axis=1).astype(float), columns=['out_degree']))

        # 이제 레이어별로 in/out degree 를 join 하자
        degree_per_layer = []
        for indf, outdf in zip(in_degrees, out_degrees):
            df = indf.join(outdf, how='outer')
            df['degree'] = df['in_degree'] + df['out_degree']
            degree_per_layer.append(df)

        # 그리고 레이어별 DataFrame 을 합친다
        degree_df = pd.concat(degree_per_layer, axis=0)

        # 4) merge importance + degree via concat
        importance_degree_df = pd.concat([layer_importance_df, degree_df], axis=1)

        # 5) connectivity‐corrected score
        scored = connectivity_corrected_scores(importance_degree_df)

        # 6) save per-seed results
        scored.to_csv(out_csv)
        print(f"Seed {seed} → saved {out_csv}")

if __name__ == "__main__":
    main()
