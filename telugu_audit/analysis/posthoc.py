from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from telugu_audit.analysis.fertility_accuracy_corr import correlate_fertility_accuracy
from telugu_audit.analysis.script_gap_stats import per_sentence_gap, test_gap_significance
from telugu_audit.corpus.loaders import load_corpus
from telugu_audit.corpus.schema import (
    get_allowed_registers,
    get_language_profiles,
)
from telugu_audit.metrics.fertility import get_word_count_fn
from telugu_audit.reporting.plots import plot_fertility_accuracy_scatter
from telugu_audit.run_utils import load_yaml_config, resolve_tokenizer_include
from telugu_audit.tokenizers.registry import load_tokenizers


def run_script_gap_significance_stage(
    experiment_dir: str | Path,
    config: dict,
    config_path: str | Path,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Path:
    """Write paired native-vs-romanized significance results for the active tokenizers."""
    exp_dir = Path(experiment_dir)
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)

    config_path = Path(config_path)
    word_count_method = config.get("primary_word_count_method", "whitespace")
    word_count_fn = get_word_count_fn(word_count_method)
    include = resolve_tokenizer_include(config, config_path)
    tokenizers = load_tokenizers(include=include)

    corpus_dir = config["corpus_dir"]
    allowed_registers = get_allowed_registers(config)
    significance_results: dict[str, dict[str, dict]] = {}
    gap_frames: list[pd.DataFrame] = []

    for profile in get_language_profiles(config):
        native_lines = load_corpus(
            corpus_dir,
            profile.native_informal_register,
            allowed_registers=allowed_registers,
        )
        romanized_lines = load_corpus(
            corpus_dir,
            profile.romanized_informal_register,
            allowed_registers=allowed_registers,
        )
        language_results: dict[str, dict] = {}

        for name, count_fn in tokenizers.items():
            gap_df = per_sentence_gap(
                native_lines,
                romanized_lines,
                count_fn,
                word_count_fn=word_count_fn,
            )
            gap_df.insert(0, "language", profile.name)
            gap_df.insert(1, "language_code", profile.code)
            gap_df.insert(2, "native_register", profile.native_informal_register)
            gap_df.insert(3, "romanized_register", profile.romanized_informal_register)
            gap_df.insert(4, "tokenizer", name)
            gap_df.insert(5, "word_count_method", word_count_method)
            gap_frames.append(gap_df)

            try:
                stats = test_gap_significance(
                    gap_df, n_bootstrap=n_bootstrap, seed=seed
                )
                stats["word_count_method"] = word_count_method
                stats["native_register"] = profile.native_informal_register
                stats["romanized_register"] = profile.romanized_informal_register
                language_results[name] = stats
            except ValueError as exc:
                language_results[name] = {"error": str(exc)}

        significance_results[profile.name] = language_results

    sig_path = results_dir / "script_gap_significance.json"
    sig_path.write_text(json.dumps(significance_results, indent=2), encoding="utf-8")

    if gap_frames:
        pd.concat(gap_frames, ignore_index=True).to_csv(
            results_dir / "per_sentence_gaps.csv", index=False
        )

    return sig_path


def run_fertility_accuracy_correlation_stage(
    experiment_dir: str | Path,
    benchmark_scores_path: str | Path,
    register: str,
) -> Path:
    """Correlate fertility against benchmark accuracy and emit JSON + scatter plot."""
    exp_dir = Path(experiment_dir)
    fertility_df = pd.read_csv(exp_dir / "results" / "fertility_by_register.csv")
    subset = fertility_df[fertility_df["register"] == register]
    fertility_by_model = dict(
        zip(subset["tokenizer"], subset["fertility_tokens_per_word"])
    )

    with Path(benchmark_scores_path).open(encoding="utf-8") as f:
        benchmark_cfg = yaml.safe_load(f)

    accuracy_by_model = benchmark_cfg.get("accuracy_by_model", {})
    if not accuracy_by_model:
        raise ValueError(
            "benchmark-scores YAML must contain accuracy_by_model - "
            "supply cited benchmark numbers"
        )

    result = correlate_fertility_accuracy(fertility_by_model, accuracy_by_model)
    result["register"] = register
    out_path = exp_dir / "results" / "fertility_accuracy_correlation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    plot_fertility_accuracy_scatter(
        fertility_by_model,
        accuracy_by_model,
        exp_dir / "figures",
    )
    return out_path


def load_config_for_stage(config_path: str | Path) -> dict:
    return load_yaml_config(Path(config_path))
