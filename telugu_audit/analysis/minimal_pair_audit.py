from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from telugu_audit.corpus.loaders import load_minimal_pairs
from telugu_audit.run_utils import resolve_tokenizer_include
from telugu_audit.tokenizers.registry import load_tokenizers

logger = logging.getLogger(__name__)


def run_minimal_pair_stage(
    experiment_dir: Path,
    config: dict,
    config_path: Path,
) -> None:
    """Tokenize each minimal-pair word with every active tokenizer."""
    include = resolve_tokenizer_include(config, config_path)
    tokenizers = load_tokenizers(include=include)
    pairs = load_minimal_pairs(config["minimal_pairs_path"])

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in pairs.iterrows():
        word = row["word"]
        for name, count_fn in tokenizers.items():
            try:
                n_tokens = count_fn(word)
            except Exception as exc:
                logger.warning(
                    "Tokenizer %r failed on word %r, skipping: %s", name, word, exc
                )
                continue
            rows.append(
                {
                    "word": word,
                    "gloss": row["gloss"],
                    "morph_type": row["morph_type"],
                    "tokenizer": name,
                    "n_tokens": n_tokens,
                }
            )

    pd.DataFrame(rows).to_csv(results_dir / "minimal_pair_fertility.csv", index=False)

