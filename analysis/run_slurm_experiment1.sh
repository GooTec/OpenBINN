#!/bin/bash
# Run experiment1.py with slurm using the openBINN environment.
# Usage: bash run_slurm_experiment1.sh --device gpu

set -e

PY="$HOME/.conda/envs/openBINN/bin/python"
DEVICE="gpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"; shift 2;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

if [[ "$DEVICE" == "gpu" ]]; then
  SRUN_PREFIX="srun --gres=gpu:A4000 -p gpu --time=50:00:00"
else
  SRUN_PREFIX="srun --time=50:00:00"
fi

$SRUN_PREFIX "$PY" analysis/experiment1.py

