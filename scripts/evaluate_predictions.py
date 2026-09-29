#!/usr/bin/env python3
"""Evaluate generated questions with text-overlap, lexical, and answerability metrics.

Input JSONL format per line:
{
  "context": "...",
  "answer": "...",
  "reference_question": "...",
  "generated_question": "..."
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from primal_dual_qg.metrics import exact_match_score, token_f1_score, lexical_novelty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="JSONL predictions file")
    parser.add_argument("--qa-model", default="distilbert/distilbert-base-cased-distilled-squad")
    parser.add_argument("--skip-qa", action="store_true", help="Skip answerability/consistency model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No prediction rows found")

    generated = [row["generated_question"] for row in rows]
    references = [row["reference_question"] for row in rows]

    bleu = sacrebleu.corpus_bleu(generated, [references]).score
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = np.mean([
        rouge.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(generated, references)
    ])
    novelty = np.mean([
        lexical_novelty(row["generated_question"], row["context"], row.get("answer", ""))
        for row in rows
    ])

    metrics = {
        "num_examples": len(rows),
        "bleu": bleu,
        "rougeL_f1": float(rouge_l),
        "lexical_novelty": float(novelty),
    }

    if not args.skip_qa:
        qa = pipeline("question-answering", model=args.qa_model)
        em_scores = []
        f1_scores = []
        for row in rows:
            result = qa(question=row["generated_question"], context=row["context"])
            predicted_answer = result["answer"]
            em_scores.append(exact_match_score(predicted_answer, row["answer"]))
            f1_scores.append(token_f1_score(predicted_answer, row["answer"]))
        metrics["answer_consistency_em"] = float(np.mean(em_scores))
        metrics["answer_consistency_f1"] = float(np.mean(f1_scores))

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
