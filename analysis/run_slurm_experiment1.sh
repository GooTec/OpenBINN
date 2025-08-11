#!/bin/bash
#SBATCH -J experiment1
#SBATCH -p gpu
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=50:00:00
#SBATCH --output=exp1-%j.out
#SBATCH --error=exp1-%j.err

# 1) 환경 초기화
source ~/.bashrc          # 여기까지는 -u 없이

# 2) 엄격 모드 ON
set -euxo pipefail        # 이제부터 unbound 변수 검사

# 3) Conda 환경
conda activate openBINN
which python              # 경로 확인(디버그용)

# 4) 경로 설정
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# 5) 실행
beta=2.0
gamma=2.0
rep=1

data_dir="data/experiment1/b${beta}_g${gamma}/${rep}"
results_dir="results/experiment1/b${beta}_g${gamma}/${rep}"
mkdir -p "$results_dir"

python analysis/experiment1.py
python analysis/model_comparison.py --data-dir "$data_dir" --output-dir "$results_dir/comparison"

# create symlinks so importance calculation can locate data and trained model
ln -sfn "data/experiment1/b${beta}_g${gamma}" "data/b${beta}_g${gamma}"
ln -sfn "$(readlink -f "$results_dir")" "$data_dir/results"

# compute BINN explanations and summarize per-gene importances
python analysis/importance_calculation.py --start_sim "$rep" --end_sim "$rep" --beta "$beta" --gamma "$gamma"
python analysis/feature_importance_summary.py --data-dir "$data_dir" --binn-dir "$data_dir" --out-dir "$results_dir/importance_summary"
