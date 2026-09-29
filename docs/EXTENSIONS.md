# Extension Roadmap

This file separates low-cost portfolio engineering from experiments that require model training.

## Tier 1 — portfolio engineering (no full retraining required)

- Keep the historical/corrected primal-dual implementation intact.
- Add a clean answer-aware BART baseline.
- Add a reusable evaluation script with BLEU, ROUGE-L, lexical novelty, and QA-based answer-consistency metrics.
- Add lightweight unit tests and GitHub Actions CI.
- Add an optional Gradio demo that loads a local fine-tuned checkpoint.
- Save prediction files as JSONL so results are reproducible and inspectable.

## Tier 2 — small reproducible experiment

Train the BART baseline on a bounded SQuAD subset (for example 5k–20k training samples), then evaluate both the baseline and the corrected primal-dual model on the same held-out slice.

Recommended result table:

| Model | BLEU | ROUGE-L | QA EM | QA F1 | Lexical novelty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Answer-aware BART baseline |  |  |  |  |  |
| Corrected primal-dual |  |  |  |  |  |

Do not compare scores unless preprocessing and evaluation sets are identical.

## Tier 3 — ablation study

Expose the loss weights and disable individual objectives:

- QG only: `alpha=0`, `beta=0`
- QG + QA: `alpha>0`, `beta=0`
- QG + uncommon-word distillation: `alpha=0`, `beta>0`
- Full model: `alpha>0`, `beta>0`

This is the cleanest way to show whether the dual QA loss and uncommon-word objective add measurable value.

## Tier 4 — stronger research extensions

### Answerability / cycle consistency

Generate a question, answer it with an independent extractive QA model using the source passage, and compare the predicted answer with the target answer. This directly measures whether the generated question is answerable by the intended answer.

### Explicit uncommon-word control

Turn lexical rarity into a controllable generation variable (for example low / medium / high novelty) and measure the trade-off between rarity, fluency, and answer consistency.

### Cross-domain generalization

Train on SQuAD and evaluate on another extractive QA corpus after adapting only the data loader. Report the change in answer-consistency and lexical-novelty metrics rather than claiming direct comparability to SQuAD BLEU.

### Difficulty control

Estimate question difficulty using answerability confidence, question length, lexical rarity, or a learned difficulty model. Add a requested difficulty level as a control token during generation.

### Calibration and uncertainty

Report beam-score margins, generation entropy, or QA confidence so the system can flag low-confidence generated questions rather than presenting every output as equally reliable.

### Human evaluation

For a small random sample, score grammaticality, answerability, relevance, and novelty on a 1–5 rubric. Keep the annotation sheet in `artifacts/` and report inter-annotator agreement if more than one evaluator is available.

## Tier 5 — productization

- Publish the BART baseline checkpoint to Hugging Face Hub if licensing/storage allow.
- Host `demo/app.py` on Hugging Face Spaces.
- Add example outputs to the README.
- Add a model card documenting training data, intended use, limitations, and known failure modes.

The research model should not be described as production-ready unless it has been independently rerun and evaluated after the implementation corrections.
