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

## Numerical-stability audit after the first restored ablation run

The first four-way ablation campaign completed at the Slurm level but produced `nan`/`inf` validation losses and no checkpoints. That run exposed additional bugs in the intermediate restoration and is not reported as a research result.

The final stable trainer fixes the following issues:

6. **Gold end position could be masked to `-inf`.** The earlier QA head constrained end positions using the model's *predicted* start before computing supervised cross-entropy. If the predicted start fell after the gold end, the target end logit became `-inf`, forcing infinite QA loss. The stable trainer teacher-forces a valid gold start for end-feature construction during training and masks only non-passage positions in the end logits.
7. **Zero-weight losses could still contaminate the total.** In floating-point arithmetic, `0 * nan` is still `nan`. The final trainer completely skips inactive QA/KD branches for the corresponding ablations instead of computing them and multiplying by zero.
8. **Distillation used numerically fragile `log(softmax(.))`.** The final implementation uses temperature-smoothed `log_softmax`/`softmax` with `T=2.0`, matching the paper's modified cross-entropy motivation while avoiding probability underflow.
9. **The distillation mask was used as an attention mask rather than masking input tokens.** The final implementation replaces selected valid tokens with BART's mask token and keeps a normal padding attention mask. It selects approximately 10% of non-special, non-padding tokens.
10. **SQuAD character offsets could be shifted by stripping the context.** The final preprocessing preserves the original context string exactly when mapping `answer_start` offsets to token positions.
11. **Teacher parameters are excluded from checkpoints.** The pretrained distillation teacher is frozen and reconstructed from the documented base model, avoiding redundant multi-hundred-megabyte checkpoint payloads.

These changes are intentionally documented because the purpose of the restored repository is to show both the original modeling idea and the engineering/research audit that made the experiments defensible.
