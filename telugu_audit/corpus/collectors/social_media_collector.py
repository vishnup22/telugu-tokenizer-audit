"""Collect native-informal Telugu sentences from social media datasets.

Sources:
  mounikaiiith/Telugu_Sentiment    (CC-BY-4.0) — ~35K sentences
  mounikaiiith/Telugu-Hatespeech   (CC-BY-4.0) — ~35K sentences

Both datasets are social media text (news comments, Twitter) in native Telugu
script, cited in Marreddy et al. 2022 (arXiv:2205.01204).

Register: native_informal
"""

from __future__ import annotations

import random
from pathlib import Path

_SOURCES = [
    ("mounikaiiith/Telugu_Sentiment", "Sentence"),
    ("mounikaiiith/Telugu-Hatespeech", "Sentence"),
]


def collect_social_media(
    output_path: str | Path,
    n_samples: int = 1000,
    seed: int = 42,
) -> int:
    """Sample informal Telugu sentences and write to output_path.

    Returns the number of lines written.
    """
    from datasets import concatenate_datasets, load_dataset

    all_lines: list[str] = []
    for dataset_id, text_col in _SOURCES:
        splits = ["train", "validation", "test"]
        ds = concatenate_datasets(
            [load_dataset(dataset_id, split=s) for s in splits]
        )
        for example in ds:
            text = example[text_col]
            if isinstance(text, str):
                text = text.strip()
                if 10 <= len(text) <= 500:
                    all_lines.append(text)

    # deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for line in all_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    rng = random.Random(seed)
    sampled = rng.sample(unique, min(n_samples, len(unique)))

    Path(output_path).write_text("\n".join(sampled) + "\n", encoding="utf-8")
    return len(sampled)
