# Reproducible Results

Record the exact data slice, random seed, checkpoint, and command used for every reported number.

## Experiment metadata

- Date:
- Git commit:
- Hardware:
- Dataset / split:
- Number of examples:
- Random seed:
- Model checkpoint:
- Generation settings:

## Metrics

| Model | BLEU | ROUGE-L | Answer consistency EM | Answer consistency F1 | Lexical novelty |
| --- | ---: | ---: | ---: | ---: | ---: |
| BART baseline |  |  |  |  |  |
| QG-only ablation |  |  |  |  |  |
| QG + QA |  |  |  |  |  |
| QG + uncommon-word objective |  |  |  |  |  |
| Full corrected model |  |  |  |  |  |

## Notes

- Report historical class-submission metrics separately from corrected reruns.
- Do not mix metrics from different preprocessing pipelines or evaluation subsets.
