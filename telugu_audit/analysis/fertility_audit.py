from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from telugu_audit.analysis.token_distribution import run_distribution_stage
from telugu_audit.corpus.loaders import REGISTERS, load_all_registers
from telugu_audit.metrics.fertility import compute_fertility, get_word_count_fn
from telugu_audit.run_utils import resolve_tokenizer_include
from telugu_audit.tokenizers.registry import load_tokenizers

logger = logging.getLogger(__name__)


def _resolve_word_count_methods(config: dict) -> tuple[str, dict[str, object]]:
    primary = config.get("primary_word_count_method", "whitespace")
    requested = config.get("word_count_methods", [primary])

    methods: dict[str, object] = {}
    for method_name in requested:
        try:
            methods[method_name] = get_word_count_fn(method_name)
        except ImportError as exc:
            logger.warning("Skipping word-count method %r: %s", method_name, exc)
        except ValueError as exc:
            logger.warning("Skipping word-count method %r: %s", method_name, exc)

    if primary not in methods:
        raise RuntimeError(
            f"Primary word-count method {primary!r} is unavailable; "
            f"loaded methods: {sorted(methods)}"
        )
    return primary, methods


def _build_gap_rows(fertility_df: pd.DataFrame) -> list[dict]:
    gap_rows = []
    for (name, method), tok_df in fertility_df.groupby(["tokenizer", "word_count_method"]):
        native = tok_df[tok_df["register"] == "native_informal"]
        roman = tok_df[tok_df["register"] == "romanized_informal"]
        if native.empty or roman.empty:
            continue
        native_f = native.iloc[0]["fertility_tokens_per_word"]
        roman_f = roman.iloc[0]["fertility_tokens_per_word"]
        ratio = native_f / roman_f if roman_f else float("nan")
        gap_rows.append(
            {
                "tokenizer": name,
                "word_count_method": method,
                "native_fertility": native_f,
                "romanized_fertility": roman_f,
                "script_fertility_ratio_native_over_romanized": ratio,
            }
        )
    return gap_rows


def _write_rank_stability(
    sensitivity_df: pd.DataFrame,
    primary_method: str,
    output_path: Path,
) -> None:
    summary: dict[str, dict] = {}
    primary_df = sensitivity_df[sensitivity_df["word_count_method"] == primary_method]

    for register in sorted(sensitivity_df["register"].unique()):
        register_summary = {"reference_method": primary_method, "comparisons": {}}
        baseline = primary_df[primary_df["register"] == register].sort_values(
            "fertility_tokens_per_word"
        )
        baseline_order = baseline["tokenizer"].tolist()
        for method in sorted(sensitivity_df["word_count_method"].unique()):
            comparison = sensitivity_df[
                (sensitivity_df["register"] == register)
                & (sensitivity_df["word_count_method"] == method)
            ].sort_values("fertility_tokens_per_word")
            register_summary["comparisons"][method] = {
                "stable_rank_order": comparison["tokenizer"].tolist() == baseline_order,
                "tokenizer_order": comparison["tokenizer"].tolist(),
            }
        summary[register] = register_summary

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_fertility_stage(
    experiment_dir: Path,
    config: dict,
    config_path: Path,
) -> dict[str, int]:
    """Compute per-register fertility and script-fairness gap."""
    include = resolve_tokenizer_include(config, config_path)
    tokenizers = load_tokenizers(include=include)
    if not tokenizers:
        raise RuntimeError("No tokenizers loaded - check config and API keys")
    primary_word_count_method, word_count_methods = _resolve_word_count_methods(config)

    corpora = load_all_registers(config["corpus_dir"])
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    fertility_rows = []
    for name, count_fn in tokenizers.items():
        for method_name, word_count_fn in word_count_methods.items():
            for register, lines in corpora.items():
                stats = compute_fertility(
                    count_fn,
                    lines,
                    word_count_fn=word_count_fn,
                    word_count_method=method_name,
                )
                fertility_rows.append({"tokenizer": name, "register": register, **stats})

    sensitivity_df = pd.DataFrame(fertility_rows)
    sensitivity_df.to_csv(
        results_dir / "fertility_by_register_sensitivity.csv", index=False
    )
    primary_df = sensitivity_df[
        sensitivity_df["word_count_method"] == primary_word_count_method
    ].reset_index(drop=True)
    primary_df.to_csv(results_dir / "fertility_by_register.csv", index=False)

    gap_sensitivity_df = pd.DataFrame(_build_gap_rows(sensitivity_df))
    gap_sensitivity_df.to_csv(
        results_dir / "script_fairness_gap_sensitivity.csv", index=False
    )
    gap_primary_df = gap_sensitivity_df[
        gap_sensitivity_df["word_count_method"] == primary_word_count_method
    ].reset_index(drop=True)
    gap_primary_df.to_csv(results_dir / "script_fairness_gap.csv", index=False)

    _write_rank_stability(
        sensitivity_df,
        primary_word_count_method,
        results_dir / "fertility_rank_stability.json",
    )

    run_distribution_stage(
        experiment_dir,
        corpora,
        tokenizers,
        word_count_fn=word_count_methods[primary_word_count_method],
        word_count_method=primary_word_count_method,
    )

    return {r: len(corpora[r]) for r in REGISTERS if r in corpora}
