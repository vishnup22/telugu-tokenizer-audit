"""Collect native-formal Telugu sentences from Wikipedia via Hugging Face.

Source: vengi-ai/telugu-wikipedia-clean (CC-BY-SA-4.0, derived from Telugu Wikipedia)
Register: native_formal
"""

from __future__ import annotations

import random
import re
from pathlib import Path


def _split_sentences(text: str) -> list[str]:
    """Split a Wikipedia article body into sentence-like units."""
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r"(?<=[.।?!])\s+", paragraph)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def collect_wikipedia(
    output_path: str | Path,
    n_samples: int = 1000,
    seed: int = 42,
) -> int:
    """Sample sentences from Telugu Wikipedia and write to output_path.

    Returns the number of lines written.
    """
    from datasets import load_dataset

    ds = load_dataset("vengi-ai/telugu-wikipedia-clean", split="train")

    candidates: list[str] = []
    for example in ds:
        for sentence in _split_sentences(example["text"]):
            if 30 <= len(sentence) <= 400:
                candidates.append(sentence)

    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(n_samples, len(candidates)))

    Path(output_path).write_text("\n".join(sampled) + "\n", encoding="utf-8")
    return len(sampled)
