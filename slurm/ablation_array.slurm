#!/bin/bash
#SBATCH --job-name=pdqg-ablate
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/experiments/ablation-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${PDQG_CONDA_ENV:-base}"
conda activate "$CONDA_ENV"
names=(qg_only qg_qa qg_uw full)
nvidia-smi
python scripts/run_ablation_matrix.py --only "${names[$SLURM_ARRAY_TASK_ID]}" --epochs 1 --batch-size 4 --max-train-samples 5000 --max-eval-samples 500
