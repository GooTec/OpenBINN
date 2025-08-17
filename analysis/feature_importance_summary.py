#!/usr/bin/env python
"""Summarize gene importance across models from stored explanations.

The script reads logistic-regression coefficients, FCNN attribution CSVs, and
BINN explanations and produces a gene-level summary comparing estimated and
true importances.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd

METHODS = ["ig", "shap", "deeplift", "deepliftshap"]


def summarize_logistic(exp_dir: Path) -> pd.Series:
    fp = exp_dir / "results" / "explanations" / "Logistic" / "logistic_beta.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    return df.set_index("gene")["importance"]


def summarize_fcnn(exp_dir: Path) -> dict[str, pd.Series]:
    summary: dict[str, pd.Series] = {}
    exp_root = exp_dir / "results" / "explanations" / "FCNN"
    for method in METHODS:
        files = sorted(exp_root.glob(f"FCNN_*_{method}_L1_layer0_test.csv"))
        if not files:
            continue
        df = pd.read_csv(files[0])
        gene_cols = [c for c in df.columns if c not in {"sample_id", "label", "prediction"}]
        summary[method] = df[gene_cols].sum(0)

    return summary


def summarize_binn(exp_dir: Path) -> dict[str, dict[int, pd.Series]]:
    """Collect gene importances for each PNET output layer individually."""
    summaries: dict[str, dict[int, pd.Series]] = {}
    exp_root = exp_dir / "results" / "explanations" / "PNET"
    for method in METHODS:
        layer_files = sorted(exp_root.glob(f"PNet_*_{method}_L*_layer0_test.csv"))
        if not layer_files:
            continue
        per_layer: dict[int, pd.Series] = {}
        for fp in layer_files:
            match = re.search(r"_L(\d+)_", fp.name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            df = pd.read_csv(fp)
            gene_cols = [c for c in df.columns if c not in {"sample_id", "label", "prediction"}]
            per_layer[layer_idx] = df[gene_cols].sum(0)
        summaries[method] = per_layer
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize gene importances")
    ap.add_argument("--data-dir", required=True, help="Simulation data directory")
    ap.add_argument("--out-dir", default="importance_summary", help="Output directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    log_imp = summarize_logistic(data_dir)

    gt_fp = data_dir / "ground_truth.json"
    true_genes: set[str]
    if gt_fp.exists():
        with gt_fp.open() as f:
            gt = json.load(f)
        true_genes = set(gt.get("true_genes", []))
    else:
        csv_fp = data_dir / "true_genes.csv"
        if not csv_fp.exists():
            raise FileNotFoundError("ground_truth.json or true_genes.csv not found in data directory")
        true_genes = set(pd.read_csv(csv_fp)["gene"])

    genes = log_imp.index
    truth = pd.Series([1 if g in true_genes else 0 for g in genes], index=genes)

    log_imp = log_imp.reindex(genes).fillna(0)
    fcnn_imp = summarize_fcnn(data_dir)
    binn_imp = summarize_binn(data_dir)

    df = pd.DataFrame({"gene": genes, "true_gene": truth.values, "logistic": log_imp.values})
    for method, series in fcnn_imp.items():
        df[f"fcnn_{method}"] = series.reindex(genes).fillna(0).values
    for method, layer_dict in binn_imp.items():
        for layer, series in layer_dict.items():
            df[f"binn_{method}_L{layer}"] = series.reindex(genes).fillna(0).values
    df.to_csv(out_dir / "gene_importance_summary.csv", index=False)

    import matplotlib.pyplot as plt

    def save_scatter(col: str, label: str) -> None:
        plt.figure(figsize=(6, 4))
        plt.scatter(df["true_gene"], df[col], alpha=0.7)
        plt.xlabel("True important gene")
        plt.ylabel("Estimated importance")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(out_dir / f"{label}_scatter.png")

    # logistic regression
    save_scatter("logistic", "logistic")

    # FCNN methods
    for m in METHODS:
        col = f"fcnn_{m}"
        if col in df.columns:
            save_scatter(col, col)

    # PNET methods per output layer
    for m in METHODS:
        layer_dict = binn_imp.get(m, {})
        for layer in sorted(layer_dict):
            col = f"binn_{m}_L{layer}"
            if col in df.columns:
                save_scatter(col, col)


if __name__ == "__main__":
    main()
