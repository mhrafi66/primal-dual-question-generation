# Research audit

This repository intentionally separates the historical class submission from the corrected portfolio implementation.

## Confirmed historical issues

1. **Ground-truth-question leakage in the dual QA branch.** The submitted QA preprocessing consumed the reference question, so QA loss did not test whether the generated question recovered the target answer. The restored implementation couples QA to the QG decoder representation instead.
2. **Checkpoint restoration in evaluation.** The submitted generation script loaded a checkpoint file but did not restore the model state before decoding. The corrected script loads the saved state dict.
3. **Beam accounting.** The historical decoder ranked hypotheses using the newest token score rather than accumulated beam score; this was corrected.
4. **Answer span alignment / one-token answers.** Character-to-token mapping and end-span masking were corrected.
5. **Padding in generation loss.** Padding tokens are ignored in the corrected QG loss.

The original scripts remain in `legacy/` so these changes are auditable rather than hidden.

## Important architectural caveat

The report's architecture figure labels the knowledge-distillation teacher as a pretrained T5 model, while the submitted implementation uses BART-family components. The restored code preserves the implementation lineage instead of claiming an exact paper reproduction. Any future paper-faithful reproduction should make the teacher/model choice explicit and evaluate that choice separately.

## Evaluation policy

The historical BLEU score is retained as a class-submission artifact, but new portfolio results should only be reported when produced from a versioned checkpoint and a saved prediction file. Use `docs/RESULTS_TEMPLATE.md` and the scripts under `scripts/` for reproducibility.
