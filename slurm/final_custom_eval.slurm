#!/bin/bash
#SBATCH --job-name=pdqg-final-eval
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/experiments/final-eval-%j.out
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${PDQG_CONDA_ENV:-pdqg_hpc}"

mkdir -p artifacts/experiments
nvidia-smi

# Fair 500-example BART baseline slice from the already completed 1000-example run.
if [[ ! -f artifacts/experiments/bart_predictions.jsonl ]]; then
  python scripts/generate_dataset_predictions.py \
    --model outputs/bart-answer-aware \
    --output artifacts/experiments/bart_predictions.jsonl \
    --dataset squad --max-samples 1000
fi
head -n 500 artifacts/experiments/bart_predictions.jsonl > artifacts/experiments/bart_baseline_500_predictions.jsonl
python scripts/evaluate_predictions.py artifacts/experiments/bart_baseline_500_predictions.jsonl \
  | tee artifacts/experiments/bart_baseline_500_metrics.json
python scripts/analyze_uncommon_words.py artifacts/experiments/bart_baseline_500_predictions.jsonl \
  --max-frequency 5 --max-corpus-samples 20000 \
  | tee artifacts/experiments/bart_baseline_500_uncommon.json
python scripts/merge_metric_files.py \
  --metrics artifacts/experiments/bart_baseline_500_metrics.json \
  --uncommon artifacts/experiments/bart_baseline_500_uncommon.json \
  --run bart_baseline_500 \
  --output artifacts/experiments/bart_baseline_500_combined.json

for name in qg_only qg_qa qg_uw full; do
  echo "=== $name SQuAD ==="
  python scripts/generate_primal_dual_predictions.py \
    --checkpoint "checkpoints/${name}.pth" \
    --output "artifacts/experiments/${name}_predictions.jsonl" \
    --dataset squad --max-samples 500 --num-beams 4 --max-new-tokens 48

  python scripts/evaluate_predictions.py "artifacts/experiments/${name}_predictions.jsonl" \
    | tee "artifacts/experiments/${name}_metrics.json"
  python scripts/analyze_uncommon_words.py "artifacts/experiments/${name}_predictions.jsonl" \
    --max-frequency 5 --max-corpus-samples 20000 \
    | tee "artifacts/experiments/${name}_uncommon.json"
  python scripts/merge_metric_files.py \
    --metrics "artifacts/experiments/${name}_metrics.json" \
    --uncommon "artifacts/experiments/${name}_uncommon.json" \
    --run "$name" \
    --output "artifacts/experiments/${name}_combined.json"
done

# Cross-domain evaluation for the full restored model on the same 500 HotpotQA examples
# already used for the BART baseline when available.
if [[ ! -f artifacts/hotpotqa_eval.jsonl ]]; then
  python scripts/prepare_hotpotqa.py --output artifacts/hotpotqa_eval.jsonl --max-samples 500
fi
python scripts/generate_primal_dual_predictions.py \
  --checkpoint checkpoints/full.pth \
  --dataset jsonl \
  --input-jsonl artifacts/hotpotqa_eval.jsonl \
  --output artifacts/experiments/full_hotpot_predictions.jsonl \
  --max-samples 500 --num-beams 4 --max-new-tokens 48
python scripts/evaluate_predictions.py artifacts/experiments/full_hotpot_predictions.jsonl \
  | tee artifacts/experiments/full_hotpot_metrics.json
python scripts/analyze_uncommon_words.py artifacts/experiments/full_hotpot_predictions.jsonl \
  --max-frequency 5 --max-corpus-samples 20000 \
  | tee artifacts/experiments/full_hotpot_uncommon.json
python scripts/merge_metric_files.py \
  --metrics artifacts/experiments/full_hotpot_metrics.json \
  --uncommon artifacts/experiments/full_hotpot_uncommon.json \
  --run full_hotpot \
  --output artifacts/experiments/full_hotpot_combined.json

# Merge the already-completed BART HotpotQA run into the same report format.
if [[ -f artifacts/experiments/bart_hotpot_metrics.json && -f artifacts/experiments/bart_hotpot_uncommon.json ]]; then
  python scripts/merge_metric_files.py \
    --metrics artifacts/experiments/bart_hotpot_metrics.json \
    --uncommon artifacts/experiments/bart_hotpot_uncommon.json \
    --run bart_hotpot \
    --output artifacts/experiments/bart_hotpot_combined.json
fi

python scripts/build_results_report.py

echo "PDQG_FINAL_EVALUATION_PASS"
