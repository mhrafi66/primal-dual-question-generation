#!/usr/bin/env python3
"""Measure transparent lexical novelty / uncommon-word rates on generated questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from primal_dual_qg.metrics import build_frequency_table, lexical_novelty, uncommon_word_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="JSONL predictions file")
    parser.add_argument("--max-frequency", type=int, default=5)
    parser.add_argument("--max-corpus-samples", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]

    squad = load_dataset("squad", split="train")
    if args.max_corpus_samples:
        squad = squad.select(range(min(args.max_corpus_samples, len(squad))))
    frequency = build_frequency_table(squad["question"])

    report = []
    for row in rows:
        report.append({
            "generated_question": row["generated_question"],
            "lexical_novelty": lexical_novelty(
                row["generated_question"], row["context"], row.get("answer", "")
            ),
            "uncommon_word_rate": uncommon_word_rate(
                row["generated_question"], frequency, args.max_frequency
            ),
        })

    if report:
        mean_novelty = sum(item["lexical_novelty"] for item in report) / len(report)
        mean_uncommon = sum(item["uncommon_word_rate"] for item in report) / len(report)
    else:
        mean_novelty = mean_uncommon = 0.0

    print(json.dumps({
        "num_examples": len(report),
        "mean_lexical_novelty": mean_novelty,
        "mean_uncommon_word_rate": mean_uncommon,
        "examples": report[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
