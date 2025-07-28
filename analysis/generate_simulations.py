#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper around ``sim_data_generation``.

This script keeps the original CLI used by ``experiment1.py`` but delegates the
actual dataset creation to :mod:`sim_data_generation`, which implements the
multi-omics pathway-based simulation.  The output directory follows the same
structure as before::

    data/[experiment{N}/]b{beta}_g{gamma}/{sim}

Use ``--pathway_nonlinear`` to randomly select a pathway and generate outcomes
using the quadratic score ``beta * S + gamma * S^2`` (with additional pathways
scaled by δ₁ and δ₂).  A ``pca_plot.png`` is saved showing the first two
principal components of the true genes and the outcome distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sim_data_generation as sd


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate simulation datasets")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=2.0)
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
    ap.add_argument("--alpha_sigma", type=float, default=20.0,
                    help="Stddev of gene coefficients when nonlinear")
    ap.add_argument("--prev", type=float, default=0.5,
                    help="Target prevalence for intercept calibration")
    args = ap.parse_args()

    end = args.end_sim if args.end_sim is not None else args.n_sim
    if end < args.start_sim:
        raise ValueError("end_sim must be >= start_sim")

    # configure sim_data_generation globals
    sd.BETA = args.beta
    sd.GAMMA = args.gamma
    out_root = Path("./data")
    if args.exp is not None:
        out_root = out_root / f"experiment{args.exp}"
    sd.OUT_ROOT = out_root

    sd.main(
        args.start_sim,
        end,
        pathway_nonlinear=args.pathway_nonlinear,
        alpha_sigma=args.alpha_sigma,
        prev=args.prev,
    )


if __name__ == "__main__":
    main()

