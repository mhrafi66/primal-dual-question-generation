#!/bin/bash
#SBATCH --job-name=pdqg-final-ablate
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/experiments/final-ablation-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${PDQG_CONDA_ENV:-pdqg_hpc}"

names=(qg_only qg_qa qg_uw full)
name="${names[$SLURM_ARRAY_TASK_ID]}"

echo "run=$name"
nvidia-smi
python scripts/run_ablation_matrix.py \
  --only "$name" \
  --epochs 1 \
  --batch-size 4 \
  --max-train-samples 5000 \
  --max-eval-samples 500 \
  --learning-rate 3e-5

test -f "checkpoints/${name}.pth"
grep -E "PDQG_TRAINING_PASS|BEST_VALIDATION_LOSS|Validation Loss" "artifacts/experiments/${name}.log" | tail -5

echo "PDQG_FINAL_ABLATION_PASS $name"
