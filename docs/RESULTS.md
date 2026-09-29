# Experimental Results

These are results from the restored portfolio implementation. They are **not an exact reproduction** of Wang et al. (EMNLP 2022): the repository uses BART-based components, vanilla Hugging Face SQuAD validation data, and a documented restoration of the original course implementation.

## In-domain SQuAD evaluation

| Model | α | β | N | BLEU | ROUGE-L F1 | QA EM | QA F1 | Lexical novelty | Uncommon-word rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BART answer-aware baseline | — | — | 500 | 17.6334 | 0.4439 | 0.7680 | 0.8239 | 0.2895 | 0.1513 |
| Restored QG only | 0.0000 | 0.0000 | 500 | 13.4708 | 0.3896 | 0.5060 | 0.5893 | 0.2109 | 0.1937 |
| Restored QG + QA | 0.8000 | 0.0000 | 500 | 12.0497 | 0.3678 | 0.4360 | 0.5181 | 0.1963 | 0.1929 |
| Restored QG + KD | 0.0000 | 0.1500 | 500 | 12.9952 | 0.3909 | 0.4980 | 0.5824 | 0.2221 | 0.1865 |
| Restored QG + QA + KD | 0.8000 | 0.1500 | 500 | 12.1577 | 0.3569 | 0.3960 | 0.4710 | 0.2028 | 0.1935 |

### Interpretation guardrails

- The independent-QA EM/F1 columns measure whether an external QA model recovers the intended answer from the generated question and passage.
- Lexical novelty is the fraction of generated question tokens absent from the passage/answer under the repository metric.
- The uncommon-word rate is a portfolio analysis metric based on low corpus frequency; it is **not** the paper's AGS metric.
- Compare the restored ablations against one another; do not treat the EMNLP paper's published BLEU as directly comparable because the data processing/splits and implementation differ.

## Cross-domain HotpotQA — BART baseline

- Examples: **500**
- BLEU: **4.5227**
- ROUGE-L F1: **0.2327**
- QA consistency EM/F1: **0.2340 / 0.3484**
- Lexical novelty: **0.1696**
- Uncommon-word rate: **0.1907**

## Cross-domain HotpotQA — Restored full model

- Examples: **500**
- BLEU: **4.2925**
- ROUGE-L F1: **0.2219**
- QA consistency EM/F1: **0.0800 / 0.1416**
- Lexical novelty: **0.0722**
- Uncommon-word rate: **0.1786**

## Reproducibility

The corresponding JSON metrics, configs, histories, and raw prediction files are produced under `artifacts/experiments/`. Large checkpoints and raw prediction JSONL files are intentionally ignored by Git.

