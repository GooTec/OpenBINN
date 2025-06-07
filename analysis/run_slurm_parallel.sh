#!/bin/bash
# Run permutation_analysis.py in parallel using srun.
# Usage example:

#   bash run_slurm_parallel.sh --statistical_method gene-permutation --device gpu --start 1 --end 100 --parallel 10

set -e

STAT_METHOD="gene-permutation"
DEVICE="gpu"
START=1
END=100
PARALLEL=1
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
    --parallel)
      PARALLEL="$2"; shift 2;;
    --beta)
      BETA="$2"; shift 2;;
    --gamma)
      GAMMA="$2"; shift 2;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

if [[ "$DEVICE" == "gpu" ]]; then
  SRUN_PREFIX="srun --gres=gpu:A4000 -p gpu --time=50:00:00"
else
  SRUN_PREFIX="srun --time=50:00:00"
fi

# Generate data and train originals once to obtain optimal parameters
$SRUN_PREFIX python generate_b4g4_simulations.py --beta "$BETA" --gamma "$GAMMA" \
    --start_sim "$START" --end_sim "$END"
$SRUN_PREFIX python train_original.py --beta "$BETA" --gamma "$GAMMA" \
    --start_sim "$START" --end_sim "$END"


total=$((END - START + 1))
chunk=$(( (total + PARALLEL - 1) / PARALLEL ))
count=0
for ((i=0; i<PARALLEL; i++)); do
  beg=$((START + i * chunk))
  end=$((beg + chunk - 1))
  if (( beg > END )); then break; fi
  if (( end > END )); then end=$END; fi
  echo "==== Launching simulations ${beg}-${end} ===="
  $SRUN_PREFIX python permutation_analysis.py \
    --statistical_method "$STAT_METHOD" \
    --start_sim "$beg" --end_sim "$end" \
    --beta "$BETA" --gamma "$GAMMA" &
  count=$((count+1))
done
wait

echo "All simulations launched."
