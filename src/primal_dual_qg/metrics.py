"""Lightweight metrics used by the portfolio evaluation scripts.

These utilities intentionally avoid model downloads so they can be tested in CI.
"""
from __future__ import annotations

import collections
import re
import string
from typing import Iterable


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, remove punctuation/articles, squash spaces."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        table = str.maketrans("", "", string.punctuation)
        return value.translate(table)

    value = text.lower()
    value = remove_punctuation(value)
    value = remove_articles(value)
    return " ".join(value.split())


def exact_match_score(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1_score(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = collections.Counter(prediction_tokens) & collections.Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def word_tokens(text: str) -> list[str]:
    """Simple tokenizer for transparent lexical analyses."""
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def lexical_novelty(generated_question: str, context: str, answer: str = "") -> float:
    """Fraction of generated word tokens absent from the source context + answer."""
    generated = word_tokens(generated_question)
    if not generated:
        return 0.0
    source = set(word_tokens(context) + word_tokens(answer))
    novel = sum(token not in source for token in generated)
    return novel / len(generated)


def build_frequency_table(texts: Iterable[str]) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for text in texts:
        counts.update(word_tokens(text))
    return counts


def uncommon_word_rate(
    generated_question: str,
    frequency_table: collections.Counter[str],
    max_frequency: int = 5,
) -> float:
    """Fraction of generated tokens whose corpus frequency is <= max_frequency.

    This is a transparent portfolio metric, not a claim to reproduce the paper's
    uncommon-word evaluation protocol.
    """
    generated = word_tokens(generated_question)
    if not generated:
        return 0.0
    uncommon = sum(frequency_table.get(token, 0) <= max_frequency for token in generated)
    return uncommon / len(generated)
