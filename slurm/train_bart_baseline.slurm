#!/bin/bash
#SBATCH --job-name=pdqg-bart
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/experiments/bart-%j.out
#SBATCH --export=ALL

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${PDQG_CONDA_ENV:-base}"
conda activate "$CONDA_ENV"
nvidia-smi
python scripts/train_bart_baseline.py --max-train-samples 10000 --max-eval-samples 1000 --epochs 2 --batch-size 8
