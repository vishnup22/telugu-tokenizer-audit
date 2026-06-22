from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from telugu_audit.corpus.loaders import REGISTERS, load_all_registers
from telugu_audit.metrics.fertility import compute_fertility
from telugu_audit.run_utils import resolve_tokenizer_include
from telugu_audit.tokenizers.registry import load_tokenizers

logger = logging.getLogger(__name__)


def run_fertility_stage(
    experiment_dir: Path,
    config: dict,
    config_path: Path,
) -> dict[str, int]:
    """Compute per-register fertility and script-fairness gap."""
    include = resolve_tokenizer_include(config, config_path)
    tokenizers = load_tokenizers(include=include)
    if not tokenizers:
        raise RuntimeError("No tokenizers loaded — check config and API keys")

    corpora = load_all_registers(config["corpus_dir"])
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    fertility_rows = []
    for name, count_fn in tokenizers.items():
        for register, lines in corpora.items():
            stats = compute_fertility(count_fn, lines)
            fertility_rows.append({"tokenizer": name, "register": register, **stats})

    fertility_df = pd.DataFrame(fertility_rows)
    fertility_df.to_csv(results_dir / "fertility_by_register.csv", index=False)

    gap_rows = []
    for name in tokenizers:
        native = fertility_df[
            (fertility_df["tokenizer"] == name)
            & (fertility_df["register"] == "native_informal")
        ]
        roman = fertility_df[
            (fertility_df["tokenizer"] == name)
            & (fertility_df["register"] == "romanized_informal")
        ]
        if native.empty or roman.empty:
            continue
        native_f = native.iloc[0]["fertility_tokens_per_word"]
        roman_f = roman.iloc[0]["fertility_tokens_per_word"]
        ratio = native_f / roman_f if roman_f else float("nan")
        gap_rows.append(
            {
                "tokenizer": name,
                "native_fertility": native_f,
                "romanized_fertility": roman_f,
                "script_fertility_ratio_native_over_romanized": ratio,
            }
        )

    pd.DataFrame(gap_rows).to_csv(results_dir / "script_fairness_gap.csv", index=False)

    return {r: len(corpora[r]) for r in REGISTERS}
