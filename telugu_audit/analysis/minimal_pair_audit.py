from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from telugu_audit.corpus.loaders import load_minimal_pairs
from telugu_audit.corpus.schema import get_language_profiles, get_valid_morph_types
from telugu_audit.run_utils import resolve_tokenizer_include
from telugu_audit.tokenizers.registry import load_tokenizers

logger = logging.getLogger(__name__)


def run_minimal_pair_stage(
    experiment_dir: Path,
    config: dict,
    config_path: Path,
) -> None:
    """Tokenize each language's minimal-pair words with every active tokenizer.

    Each language profile may define its own minimal_pairs_path (and therefore its
    own morph_type vocabulary); the primary language falls back to the top-level
    config["minimal_pairs_path"] if it does not set one, to stay compatible with
    single-language configs. Languages with neither are skipped.
    """
    include = resolve_tokenizer_include(config, config_path)
    tokenizers = load_tokenizers(include=include)

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for profile in get_language_profiles(config):
        minimal_pairs_path = profile.minimal_pairs_path
        if minimal_pairs_path is None and profile.primary:
            minimal_pairs_path = config.get("minimal_pairs_path")
        if minimal_pairs_path is None:
            logger.info("No minimal_pairs_path configured for %s, skipping.", profile.name)
            continue

        pairs = load_minimal_pairs(
            minimal_pairs_path, valid_morph_types=get_valid_morph_types(profile.name)
        )

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
                        "language": profile.name,
                        "language_code": profile.code,
                        "word": word,
                        "gloss": row["gloss"],
                        "morph_type": row["morph_type"],
                        "tokenizer": name,
                        "n_tokens": n_tokens,
                    }
                )

    pd.DataFrame(rows).to_csv(results_dir / "minimal_pair_fertility.csv", index=False)

