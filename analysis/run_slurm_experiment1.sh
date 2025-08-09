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

# 4) 실행
beta=2.0
gamma=2.0
rep=1

data_dir=./data/experiment1/b${beta}_g${gamma}/${rep}
results_dir=./results/experiment1/b${beta}_g${gamma}/${rep}

python experiment1.py
python model_comparison.py --data-dir "$data_dir" --output-dir "$results_dir/comparison"

