# Experiment plan

## Phase 0 — smoke tests

- `pytest -q`
- Train answer-aware BART on 500 SQuAD examples for one epoch.
- Run each primal-dual loss configuration on 100–500 examples to validate plumbing.

## Phase 1 — clean baseline

Train `facebook/bart-base` on answer + context -> question. Save predictions on a fixed SQuAD validation subset.

## Phase 2 — loss ablation

| Run | alpha (QA) | beta (UW) | Question |
| --- | ---: | ---: | --- |
| `qg_only` | 0.0 | 0.0 | generation only |
| `qg_qa` | 0.8 | 0.0 | effect of dual QA |
| `qg_uw` | 0.0 | 0.15 | effect of uncommon-word objective |
| `full` | 0.8 | 0.15 | combined model |

Evaluate each run with BLEU, ROUGE-L, independent-QA EM/F1, lexical novelty, and uncommon-word rate.

## Phase 3 — answerability filtering

Generate multiple candidate questions, answer each using an independent QA model, and reject candidates below a QA-F1 threshold. Report quality/coverage trade-off.

## Phase 4 — cross-domain generalization

Train on SQuAD and evaluate generation on a prepared HotpotQA subset without further fine-tuning. Report the change in QA consistency, overlap metrics, and lexical novelty.

## Phase 5 — manual error analysis

Annotate 100–200 examples for wrong target, unanswerable, copy-heavy, too generic, hallucinated, ungrammatical, or unnatural rare-word usage. Also score grammaticality, relevance, answerability, naturalness, and novelty from 1–5.

## Phase 6 — demo

Publish the best compact checkpoint behind `demo/app.py` locally or on Hugging Face Spaces. The demo should be presented only after a checkpoint is reproducibly evaluated.
