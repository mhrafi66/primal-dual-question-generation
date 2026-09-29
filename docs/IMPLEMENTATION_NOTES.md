# Implementation Notes and Corrections

This document separates the **original class submission** from the **corrected portfolio version**.

## 1. Gold-question leakage in the QA task

### Original behavior

The submitted preprocessing code built the QA input by concatenating the passage with the **reference question**:

```python
tokenized_input_qa = tokenizer(
    context + '</s><s>' + question,
    truncation=True,
    max_length=512,
)['input_ids']
```

The QA loss therefore optimized answer extraction from the passage conditioned on the gold question. That is ordinary multi-task QA, but it does not implement the intended primal-dual connection in which the QA branch asks the **generated** question.

### Correction

The cleaned code tokenizes only the passage for the explicit QA input. During `forward()`, the hidden representation produced by the question decoder is augmented with QA task/segment embeddings and concatenated with the passage representation before the shared encoder is applied.

This has two useful properties:

1. The QA branch no longer directly receives the gold-question token sequence.
2. QA loss has a differentiable path through the generated-question representation back into the QG branch.

The QG decoder still uses teacher forcing during supervised training, as is standard for autoregressive sequence training; the important correction is removal of the independent gold-question QA input.

## 2. Answer-span alignment

The original code estimated token positions by tokenizing text prefixes. That is fragile around subword boundaries and truncation.

The corrected code uses the tokenizer's `offset_mapping` to convert SQuAD character spans into token indices. Answers truncated from the QA context receive the ignored target `-100`.

## 3. End-index masking

The original QA code masked positions using:

```python
positions <= predicted_start
```

which makes a one-token answer impossible because the end index cannot equal the start index.

The corrected code masks only:

```python
positions < predicted_start
```

and masks question positions entirely so answers must come from the passage.

## 4. Evaluation checkpoint bug

The submitted evaluation code performed:

```python
checkpoint = torch.load(model_path)
model = QuesitonGenerationWithKnowledgeDist(...)
```

but never called `load_state_dict`.

As a result, the historical evaluation script generated with a newly constructed model rather than the saved trained parameters.

The corrected script now performs:

```python
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_param'])
```

before evaluation.

## 5. Beam-search scoring

The original beam search replaced a hypothesis score with the log-probability of only the newest token. The cleaned version accumulates log-probabilities:

```python
new_score = old_score + log(next_token_probability)
```

so beam ranking reflects the whole partial sequence.

## 6. Loss masking

Question-generation targets are padded in batches. The corrected QG cross-entropy ignores `tokenizer.pad_token_id` so padding does not contribute to the optimization objective.

## 7. Historical metrics

The final report recorded BLEU = 1.275913. The corrected repository does not reinterpret or overwrite that result. Because the original evaluation failed to restore the trained checkpoint and the QA branch used the gold question, a valid corrected BLEU score requires retraining/evaluation and is intentionally left unreported.

## 8. What was deliberately not rewritten

The corrected code keeps the original project's overall architecture and BART-based implementation recognizable. It does not claim to be a faithful reproduction of every detail in Wang et al. (2022), and it does not replace the project with a new off-the-shelf question-generation pipeline.
