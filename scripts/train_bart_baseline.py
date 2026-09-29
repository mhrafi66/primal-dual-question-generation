#!/usr/bin/env python3
"""Train a simple answer-aware BART baseline on SQuAD.

This baseline gives the research model a clean point of comparison without
changing the historical primal-dual implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/bart-base")
    parser.add_argument("--output-dir", default="outputs/bart-answer-aware")
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=66)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    dataset = load_dataset("squad")
    train = dataset["train"]
    validation = dataset["validation"]

    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples:
        validation = validation.select(range(min(args.max_eval_samples, len(validation))))

    def preprocess(batch):
        answers = [entry["text"][0] for entry in batch["answers"]]
        sources = [
            f"answer: {answer} context: {context}"
            for answer, context in zip(answers, batch["context"])
        ]
        model_inputs = tokenizer(
            sources,
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["question"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = train.column_names
    train = train.map(preprocess, batched=True, remove_columns=remove_columns)
    validation = validation.map(preprocess, batched=True, remove_columns=validation.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_eval_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}")
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            running_loss += outputs.loss.item()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="validation"):
                batch = {key: value.to(device) for key, value in batch.items()}
                eval_loss += model(**batch).loss.item()
        eval_loss /= max(len(eval_loader), 1)
        print(f"epoch={epoch + 1} validation_loss={eval_loss:.4f}")
        history.append({"epoch": epoch + 1, "validation_loss": eval_loss})
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"saved new best baseline to {output_dir}")

    import json
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"best_validation_loss={best_eval_loss:.4f}")
    print(f"saved best baseline to {output_dir}")


if __name__ == "__main__":
    main()
