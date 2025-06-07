#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""permutation_analysis.py
Run training, importance calculation and p-value estimation sequentially
for one or more simulation datasets.
"""

import argparse
import subprocess
import sys


def run(cmd):
    print("->", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(
        description="Sequential pipeline for permutation analysis"
    )
    ap.add_argument(
        "--statistical_method",
        choices=["bootstrap", "gene-permutation", "label-permutation", "all"],
        default="gene-permutation",
        help="Variant type used for training and analysis",
    )
    ap.add_argument("--start_sim", type=int, default=1)
    ap.add_argument("--end_sim", type=int, default=1)
    args = ap.parse_args()

    for i in range(args.start_sim, args.end_sim + 1):
        print(f"\n■■ Simulation {i:3d} ({args.statistical_method}) ■■")
        # Train all variants with the optimal parameters from the pre-trained
        # originals
        run([
            "python", "train_variants.py",
            "--start_sim", str(i), "--end_sim", str(i),
            "--statistical_method", args.statistical_method,
        ])
        run([
            "python", "importance_calculation.py",
            "--start_sim", str(i), "--end_sim", str(i),
            "--statistical_method", args.statistical_method,
            "--skip_original",
        ])
        run([
            "python", "pvalue_calculation.py",
            "--start_sim", str(i), "--end_sim", str(i),
            "--statistical_method", args.statistical_method,
        ])

    print("\n✓ permutation analysis finished.")


if __name__ == "__main__":
    main()
