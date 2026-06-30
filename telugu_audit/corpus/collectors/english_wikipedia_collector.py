"""Collect English Wikipedia sentences for the optional English baseline register."""

from __future__ import annotations

import random
import re
from pathlib import Path


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for part in _SENTENCE_SPLIT_RE.split(paragraph):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def collect_english_wikipedia(
    output_path: str | Path,
    n_samples: int = 1000,
    seed: int = 42,
) -> int:
    from datasets import load_dataset

    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0

    for article in ds:
        for sentence in _split_sentences(article.get("text", "")):
            word_count = len(sentence.split())
            if not 5 <= word_count <= 50:
                continue
            seen += 1
            if len(reservoir) < n_samples:
                reservoir.append(sentence)
            else:
                idx = rng.randint(0, seen - 1)
                if idx < n_samples:
                    reservoir[idx] = sentence
        if seen >= max(n_samples * 20, 5000) and len(reservoir) >= n_samples:
            break

    Path(output_path).write_text("\n".join(reservoir) + "\n", encoding="utf-8")
    return len(reservoir)
