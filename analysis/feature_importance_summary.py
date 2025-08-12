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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

METHODS = ["itg", "ig", "gradshap", "deeplift", "shap"]


def summarize_logistic(exp_dir: Path) -> pd.Series:
    fp = exp_dir / "results" / "explanations" / "Logistic" / "logistic_beta.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_csv(fp)
    return df.set_index("gene")["importance"].abs()


def summarize_fcnn(exp_dir: Path) -> dict[str, pd.Series]:
    summary: dict[str, pd.Series] = {}
    exp_root = exp_dir / "results" / "explanations" / "FCNN"
    for method in METHODS:
        files = sorted(exp_root.glob(f"FCNN_*_{method}_L1_layer0_test.csv"))
        if not files:
            continue
        df = pd.read_csv(files[0])
        gene_cols = [c for c in df.columns if c not in {"sample_id", "label", "prediction"}]
        summary[method] = df[gene_cols].abs().sum(0)
    return summary


def summarize_binn(exp_dir: Path) -> dict[str, pd.Series]:
    summary: dict[str, pd.Series] = {}
    exp_root = exp_dir / "results" / "explanations" / "PNET"
    for method in METHODS:
        layer_files = sorted(exp_root.glob(f"PNet_*_{method}_L*_layer0_test.csv"))
        if not layer_files:
            continue
        gene_imps = []
        for fp in layer_files:
            df = pd.read_csv(fp)
            gene_cols = [c for c in df.columns if c not in {"sample_id", "label", "prediction"}]
            gene_imps.append(df[gene_cols].abs().sum(0))
        summary[method] = pd.concat(gene_imps, axis=1).mean(axis=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize gene importances")
    ap.add_argument("--data-dir", required=True, help="Simulation data directory")
    ap.add_argument("--out-dir", default="importance_summary", help="Output directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    truth_fp = data_dir / "true_genes.csv"
    if not truth_fp.exists():
        raise FileNotFoundError("true_genes.csv not found in data directory")
    truth = pd.read_csv(truth_fp).set_index("gene")["important"]
    genes = truth.index

    log_imp = summarize_logistic(data_dir).reindex(genes).fillna(0)
    fcnn_imp = summarize_fcnn(data_dir)
    binn_imp = summarize_binn(data_dir)

    df = pd.DataFrame({"gene": genes, "true_gene": truth.values, "logistic": log_imp.values})
    for method, series in fcnn_imp.items():
        df[f"fcnn_{method}"] = series.reindex(genes).fillna(0).values
    for method, series in binn_imp.items():
        df[f"binn_{method}"] = series.reindex(genes).fillna(0).values
    df.to_csv(out_dir / "gene_importance_summary.csv", index=False)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.scatter(df["true_gene"], df["logistic"], label="logistic", alpha=0.7)
    if "fcnn_deeplift" in df.columns:
        plt.scatter(df["true_gene"], df["fcnn_deeplift"], label="fcnn_deeplift", alpha=0.7)
    if "binn_deeplift" in df.columns:
        plt.scatter(df["true_gene"], df["binn_deeplift"], label="binn_deeplift", alpha=0.7)
    plt.xlabel("True important gene")
    plt.ylabel("Estimated importance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gene_importance_scatter.png")


if __name__ == "__main__":
    main()
