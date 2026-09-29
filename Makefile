.PHONY: test compile baseline-smoke ablation-dry-run eval-example

compile:
	python -m py_compile QuestionAnsweringTraining.py SampleQsGenerationExampleAndBLEUScore.py src/primal_dual_qg/*.py scripts/*.py demo/app.py

test: compile
	pytest -q tests

baseline-smoke:
	python scripts/train_bart_baseline.py --max-train-samples 500 --max-eval-samples 100 --epochs 1 --batch-size 4

ablation-dry-run:
	python scripts/run_ablation_matrix.py --dry-run

eval-example:
	python scripts/evaluate_predictions.py example_predictions.jsonl --skip-qa
