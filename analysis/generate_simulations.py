#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper around ``sim_data_generation``.

This script keeps the original CLI used by ``experiment1.py`` but delegates the
actual dataset creation to :mod:`sim_data_generation`, which implements the
multi-omics pathway-based simulation.  The output directory follows the same
structure as before::

    data/[experiment{N}/]b{beta}_g{gamma}/{sim}

Use ``--pathway_nonlinear`` to randomly select a pathway and generate outcomes
using the quadratic score ``pathway_linear_effect * S + pathway_nonlinear_effect * S^2``
(with additional pathways scaled by δ₁ and δ₂).  A ``pca_plot.png`` is saved
showing the first two principal components of the true genes and the outcome
distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sim_data_generation as sd


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate simulation datasets")
    ap.add_argument("--pathway_linear_effect", "--beta", dest="pathway_linear_effect", type=float, default=2.0)
    ap.add_argument("--pathway_nonlinear_effect", "--gamma", dest="pathway_nonlinear_effect", type=float, default=2.0)
    ap.add_argument("--n_sim", type=int, default=sd.N_SIM,
                    help="Number of simulations if --end_sim is not given")
    ap.add_argument("--start_sim", type=int, default=1,
                    help="Start index of simulation (inclusive)")
    ap.add_argument("--end_sim", type=int,
                    help="End index of simulation (inclusive). Defaults to n_sim")
    ap.add_argument("--exp", type=int, default=None,
                    help="Experiment number to store data under")
    ap.add_argument("--pathway_nonlinear", action="store_true",
                    help="Use pathway-based nonlinear outcome generation")
    ap.add_argument("--gene_effect_sigma", type=float, default=20.0,
                    help="Stddev of gene coefficients when nonlinear")
    ap.add_argument("--prev", type=float, default=0.5,
                    help="Target prevalence for intercept calibration")
    args = ap.parse_args()

    end = args.end_sim if args.end_sim is not None else args.n_sim
    if end < args.start_sim:
        raise ValueError("end_sim must be >= start_sim")

    # configure sim_data_generation globals
    sd.PATHWAY_LINEAR_EFFECT = args.pathway_linear_effect
    sd.PATHWAY_NONLINEAR_EFFECT = args.pathway_nonlinear_effect
    out_root = Path("./data")
    if args.exp is not None:
        out_root = out_root / f"experiment{args.exp}"
    sd.OUT_ROOT = out_root

    sd.main(
        args.start_sim,
        end,
        pathway_nonlinear=args.pathway_nonlinear,
        gene_effect_sigma=args.gene_effect_sigma,
        prev=args.prev,
    )

    # ╭───── Logistic regression sanity check ─────╮
    def eval_dir(d: Path) -> None:
        Xm = pd.read_csv(d / "somatic_mutation_paper.csv", index_col=0)
        Xc = pd.read_csv(d / "P1000_data_CNA_paper.csv", index_col=0)
        y = pd.read_csv(d / "response.csv", index_col=0)["response"]
        cnv_del = Xc.applymap(lambda v: 1 if v == -2 else 0)
        cnv_amp = Xc.applymap(lambda v: 1 if v == 2 else 0)
        omics_effect = {"mutation": 1.0, "cnv_del": 1.0, "cnv_amp": 1.0}
        GA = (
            omics_effect["mutation"] * Xm
            + omics_effect["cnv_del"] * cnv_del
            + omics_effect["cnv_amp"] * cnv_amp
        )
        tr = pd.read_csv(d / "splits" / "training_set_0.csv", index_col=0)
        va = pd.read_csv(d / "splits" / "validation_set.csv", index_col=0)
        te = pd.read_csv(d / "splits" / "test_set.csv", index_col=0)
        model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
        model.fit(GA.loc[tr["id"]], tr["response"])
        def auc(df):
            p = model.predict_proba(GA.loc[df["id"]])[:, 1]
            return roc_auc_score(df["response"], p)
        auc_tr = auc(tr)
        auc_va = auc(va)
        auc_te = auc(te)
        pd.DataFrame({
            "train_auc": [auc_tr],
            "val_auc": [auc_va],
            "test_auc": [auc_te],
        }).to_csv(d / "logistic_metrics.csv", index=False)
        print(f"  [{d.name}] train AUC={auc_tr:.3f} val AUC={auc_va:.3f} test AUC={auc_te:.3f}")

    scen_root = out_root / f"b{args.pathway_linear_effect}_g{args.pathway_nonlinear_effect}"
    for i in range(args.start_sim, end + 1):
        eval_dir(scen_root / f"{i}")


if __name__ == "__main__":
    main()

