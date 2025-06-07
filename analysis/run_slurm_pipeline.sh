#!/bin/bash
# Run training, importance calculation and p-value estimation using srun.
# Usage: bash run_slurm_pipeline.sh --statistical_method gene-permutation --device gpu \
#        --beta 2 --gamma 2

set -e

BETA=0
GAMMA=0.0
STAT_METHOD="gene-permutation"
DEVICE="gpu"
START=1
END=100
BETA=2
GAMMA=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --statistical_method)
      STAT_METHOD="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --start)
      START="$2"; shift 2;;
    --end)
      END="$2"; shift 2;;
    --beta)
      BETA="$2"; shift 2;;
    --gamma)
      GAMMA="$2"; shift 2;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

if [[ "$DEVICE" == "gpu" ]]; then
  SRUN_PREFIX="srun --gres=gpu:A4000 -p gpu  --time=50:00:00"
else
  SRUN_PREFIX="srun  --time=50:00:00"
fi

# Generate data and train originals once
$SRUN_PREFIX python sim_data_generation.py --beta "$BETA" --gamma "$GAMMA"
$SRUN_PREFIX python train_original.py --beta "$BETA" --gamma "$GAMMA"

for ((i=START;i<=END;i++)); do
  echo "===== Simulation $i ($STAT_METHOD) ====="
  start_time=$(date +%s)
  $SRUN_PREFIX python train_variants.py --start_sim $i --end_sim $i \
      --statistical_method $STAT_METHOD \
      --beta $BETA --gamma $GAMMA
  $SRUN_PREFIX python importance_calculation.py --start_sim $i --end_sim $i \
      --statistical_method $STAT_METHOD --skip_original \
      --beta $BETA --gamma $GAMMA
  $SRUN_PREFIX python pvalue_calculation.py --start_sim $i --end_sim $i \
      --statistical_method $STAT_METHOD \
      --beta $BETA --gamma $GAMMA
  end_time=$(date +%s)
  runtime=$((end_time-start_time))
  echo "Simulation $i finished in ${runtime}s" 
  echo
done
