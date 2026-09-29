#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p artifacts/experiments checkpoints

BASE_JOB=$(sbatch --parsable \
  --mail-user=u1472438@utah.edu --mail-type=END,FAIL \
  --export=ALL,PDQG_CONDA_ENV=pdqg_hpc \
  slurm/train_bart_baseline.slurm)

echo "Best-checkpoint BART rerun: $BASE_JOB"

ABL_JOB=$(sbatch --parsable \
  --mail-user=u1472438@utah.edu --mail-type=END,FAIL \
  --export=ALL,PDQG_CONDA_ENV=pdqg_hpc \
  slurm/final_ablation_array.slurm)

echo "Fixed ablation array: $ABL_JOB"

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:${BASE_JOB}:${ABL_JOB} \
  --mail-user=u1472438@utah.edu --mail-type=END,FAIL \
  --export=ALL,PDQG_CONDA_ENV=pdqg_hpc \
  slurm/final_custom_eval.slurm)

echo "Dependent final evaluation/report: $EVAL_JOB"

cat > artifacts/experiments/final_job_ids.txt <<EOT
FINAL_BASE_JOB=$BASE_JOB
FINAL_ABL_JOB=$ABL_JOB
FINAL_EVAL_JOB=$EVAL_JOB
EOT

printf '\nSubmitted complete final campaign.\n'
printf 'BART best-checkpoint rerun: %s\nAblations: %s\nEvaluation/report: %s\n' "$BASE_JOB" "$ABL_JOB" "$EVAL_JOB"
printf 'The evaluation job will start automatically only if baseline + all four ablations succeed.\n'
squeue -j "${BASE_JOB},${ABL_JOB},${EVAL_JOB}"
