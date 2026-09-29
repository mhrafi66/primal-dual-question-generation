from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from primal_dual_qg.restored_model import MODEL_NAME, PrimalDualQuestionGenerator, build_tokenizer

SEED = 66
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER = build_tokenizer(MODEL_NAME)

MAX_TRAIN = env_int("PDQG_MAX_TRAIN_SAMPLES", 0)
MAX_EVAL = env_int("PDQG_MAX_EVAL_SAMPLES", 0)
RUN_NAME = os.environ.get("PDQG_RUN_NAME", "full")
BATCH_SIZE = env_int("PDQG_BATCH_SIZE", 32)
EPOCHS = env_int("PDQG_EPOCHS", 9)
LR = env_float("PDQG_LR", 3e-5)
ALPHA = env_float("PDQG_ALPHA", 0.8)
BETA = env_float("PDQG_BETA", 0.15)
SHUFFLE = env_bool("PDQG_SHUFFLE", True)
WARMUP = env_int("PDQG_WARMUP_STEPS", 50)
CHECKPOINT = Path(os.environ.get("PDQG_CHECKPOINT", f"checkpoints/{RUN_NAME}.pth"))
QUESTION_MAX_LENGTH = env_int("PDQG_MAX_QUESTION_LENGTH", 64)


def find_answer_token_span(offset_mapping, answer_start: int, answer_end: int):
    start_idx = None
    end_idx = None
    for idx, pair in enumerate(offset_mapping):
        start, end = pair
        if start == end:
            continue
        if start_idx is None and start <= answer_start < end:
            start_idx = idx
        if start < answer_end <= end:
            end_idx = idx
            break
    if start_idx is None or end_idx is None:
        return -100, -100
    return start_idx, end_idx


def segment_ids_from_encoding(encoding):
    seq_ids = encoding.sequence_ids()
    result = []
    last_real = 0
    for seq_id in seq_ids:
        if seq_id is None:
            result.append(last_real)
        else:
            last_real = int(seq_id)
            result.append(last_real)
    return result


def tokenize_sample(row):
    # Keep context bytes exactly aligned with SQuAD's answer_start offsets.
    context = row["context"]
    question = row["question"].strip()
    answer = row["answers"]["text"][0]
    answer_start = int(row["answers"]["answer_start"][0])
    answer_end = answer_start + len(answer)

    question_enc = TOKENIZER(
        question,
        truncation=True,
        max_length=QUESTION_MAX_LENGTH,
    )
    question_ids = question_enc["input_ids"]

    qg_enc = TOKENIZER(
        context,
        answer,
        truncation="only_first",
        max_length=512,
    )
    qg_ids = qg_enc["input_ids"]
    qg_segments = segment_ids_from_encoding(qg_enc)

    # Reserve decoder/question representation space so the combined QA sequence
    # stays within BART's 512-position limit.
    question_budget = max(1, len(question_ids) - 1)
    qa_context_max = max(8, 512 - question_budget)
    qa_enc = TOKENIZER(
        context,
        truncation=True,
        max_length=qa_context_max,
        return_offsets_mapping=True,
    )
    qa_ids = qa_enc["input_ids"]
    answer_start_tok, answer_end_tok = find_answer_token_span(
        qa_enc["offset_mapping"], answer_start, answer_end
    )

    uw_enc = TOKENIZER(context, truncation=True, max_length=512)
    uw_ids = uw_enc["input_ids"]

    return {
        "question_input_ids": question_ids,
        "qg_input_ids": qg_ids,
        "qg_task_ids": [0] * len(qg_ids),
        "qg_segment_ids": qg_segments,
        "qa_input_ids": qa_ids,
        "qa_task_ids": [1] * len(qa_ids),
        "qa_segment_ids": [0] * len(qa_ids),
        "answer_start": answer_start_tok,
        "answer_end": answer_end_tok,
        "uw_input_ids": uw_ids,
        "uw_task_ids": [2] * len(uw_ids),
        "uw_segment_ids": [0] * len(uw_ids),
    }


def pad(rows, value):
    return pad_sequence([torch.tensor(x, dtype=torch.long) for x in rows], batch_first=True, padding_value=value)


def collate(batch):
    return {
        "question_input_ids": pad([x["question_input_ids"] for x in batch], TOKENIZER.pad_token_id),
        "qg_input_ids": pad([x["qg_input_ids"] for x in batch], TOKENIZER.pad_token_id),
        "qg_task_ids": pad([x["qg_task_ids"] for x in batch], 0),
        "qg_segment_ids": pad([x["qg_segment_ids"] for x in batch], 0),
        "qa_input_ids": pad([x["qa_input_ids"] for x in batch], TOKENIZER.pad_token_id),
        "qa_task_ids": pad([x["qa_task_ids"] for x in batch], 1),
        "qa_segment_ids": pad([x["qa_segment_ids"] for x in batch], 0),
        "answer_start": torch.tensor([x["answer_start"] for x in batch], dtype=torch.long),
        "answer_end": torch.tensor([x["answer_end"] for x in batch], dtype=torch.long),
        "uw_input_ids": pad([x["uw_input_ids"] for x in batch], TOKENIZER.pad_token_id),
        "uw_task_ids": pad([x["uw_task_ids"] for x in batch], 2),
        "uw_segment_ids": pad([x["uw_segment_ids"] for x in batch], 0),
    }


def qa_span_loss(start_logits, end_logits, start_labels, end_labels):
    valid = (start_labels >= 0) & (end_labels >= 0)
    valid &= start_labels < start_logits.size(1)
    valid &= end_labels < end_logits.size(1)
    if not valid.any():
        return start_logits.new_zeros(())
    criterion = nn.CrossEntropyLoss()
    return criterion(start_logits[valid], start_labels[valid]) + criterion(end_logits[valid], end_labels[valid])


def move(batch, key):
    return batch[key].to(DEVICE, non_blocking=True)


def compute_batch(model, batch):
    compute_qa = ALPHA > 0.0
    compute_kd = BETA > 0.0
    outputs = model(
        qg_input_ids=move(batch, "qg_input_ids"),
        qg_task_ids=move(batch, "qg_task_ids"),
        qg_segment_ids=move(batch, "qg_segment_ids"),
        question_input_ids=move(batch, "question_input_ids"),
        qa_input_ids=move(batch, "qa_input_ids") if compute_qa else None,
        qa_task_ids=move(batch, "qa_task_ids") if compute_qa else None,
        qa_segment_ids=move(batch, "qa_segment_ids") if compute_qa else None,
        answer_start_positions=move(batch, "answer_start") if compute_qa else None,
        uw_input_ids=move(batch, "uw_input_ids") if compute_kd else None,
        uw_task_ids=move(batch, "uw_task_ids") if compute_kd else None,
        uw_segment_ids=move(batch, "uw_segment_ids") if compute_kd else None,
        compute_qa=compute_qa,
        compute_kd=compute_kd,
        distill_temperature=2.0,
    )

    targets = move(batch, "question_input_ids")[:, 1:]
    qg_loss = nn.functional.cross_entropy(
        outputs["qg_logits"].reshape(-1, outputs["qg_logits"].size(-1)),
        targets.reshape(-1),
        ignore_index=TOKENIZER.pad_token_id,
    )

    qa_loss = qg_loss.new_zeros(())
    if compute_qa:
        qa_loss = qa_span_loss(
            outputs["start_logits"], outputs["end_logits"],
            move(batch, "answer_start"), move(batch, "answer_end")
        )

    kd_loss = qg_loss.new_zeros(())
    if compute_kd:
        kd_loss = outputs["kd_loss"]

    total = qg_loss
    if compute_qa:
        total = total + ALPHA * qa_loss
    if compute_kd:
        total = total + BETA * kd_loss

    if not torch.isfinite(total):
        raise FloatingPointError(
            f"non-finite loss: total={float(total.detach())} "
            f"qg={float(qg_loss.detach())} qa={float(qa_loss.detach())} kd={float(kd_loss.detach())}"
        )
    return total, qg_loss.detach(), qa_loss.detach(), kd_loss.detach()


def run_epoch(model, loader, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "qg": 0.0, "qa": 0.0, "kd": 0.0}
    steps = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        bar = tqdm(loader, desc="train" if training else "validation")
        for batch in bar:
            if training:
                optimizer.zero_grad(set_to_none=True)
            total, qg, qa, kd = compute_batch(model, batch)
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            totals["total"] += float(total.detach())
            totals["qg"] += float(qg)
            totals["qa"] += float(qa)
            totals["kd"] += float(kd)
            steps += 1
            bar.set_postfix(loss=f"{totals['total']/steps:.4f}")

    return {k: v / max(steps, 1) for k, v in totals.items()}


def main():
    print("device:", DEVICE)
    print("run:", RUN_NAME, "alpha:", ALPHA, "beta:", BETA)

    train_ds = load_dataset("squad", split="train")
    eval_ds = load_dataset("squad", split="validation")
    if MAX_TRAIN:
        train_ds = train_ds.select(range(min(MAX_TRAIN, len(train_ds))))
    if MAX_EVAL:
        eval_ds = eval_ds.select(range(min(MAX_EVAL, len(eval_ds))))

    train_ds = train_ds.map(tokenize_sample, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize_sample, remove_columns=eval_ds.column_names)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=collate,
        pin_memory=torch.cuda.is_available()
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
        pin_memory=torch.cuda.is_available()
    )

    model = PrimalDualQuestionGenerator(MODEL_NAME, with_teacher=(BETA > 0.0)).to(DEVICE)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    total_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(WARMUP, max(0, total_steps - 1)),
        num_training_steps=total_steps,
    )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    exp_dir = Path("artifacts/experiments")
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": RUN_NAME,
        "device": str(DEVICE),
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "alpha": ALPHA,
        "beta": BETA,
        "max_train_samples": MAX_TRAIN,
        "max_eval_samples": MAX_EVAL,
        "checkpoint": str(CHECKPOINT),
        "seed": SEED,
        "distillation_temperature": 2.0,
    }
    (exp_dir / f"{RUN_NAME}_config.json").write_text(json.dumps(config, indent=2) + "\n")

    best = math.inf
    history = []
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler)
        val_metrics = run_epoch(model, eval_loader)
        print("Train:", train_metrics)
        print("Validation:", val_metrics)
        print("Validation Loss = ", val_metrics["total"])
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics})

        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(
                {
                    "model_param": {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                        if not k.startswith("teacher_encoder.") and not k.startswith("teacher_uw_head.")
                    },
                    "best_dev_loss": best,
                    "epoch": epoch,
                    "config": config,
                },
                CHECKPOINT,
            )
            print("saved checkpoint:", CHECKPOINT)

    (exp_dir / f"{RUN_NAME}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print("BEST_VALIDATION_LOSS=", best)
    print("PDQG_TRAINING_PASS")


if __name__ == "__main__":
    main()
