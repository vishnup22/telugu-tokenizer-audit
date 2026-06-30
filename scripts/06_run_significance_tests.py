#!/usr/bin/env python3
"""Run per-sentence significance tests on the native vs romanized script gap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.analysis.script_gap_stats import per_sentence_gap, test_gap_significance
from telugu_audit.corpus.loaders import load_corpus
from telugu_audit.metrics.fertility import get_word_count_fn
from telugu_audit.run_utils import load_yaml_config, resolve_tokenizer_include
from telugu_audit.tokenizers.registry import load_tokenizers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment-dir", required=True,
                        help="Experiment directory to write results into")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)

    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    word_count_method = config.get("primary_word_count_method", "whitespace")
    word_count_fn = get_word_count_fn(word_count_method)
    include = resolve_tokenizer_include(config, config_path)
    tokenizers = load_tokenizers(include=include)

    corpus_dir = config["corpus_dir"]
    native_lines = load_corpus(corpus_dir, "native_informal")
    romanized_lines = load_corpus(corpus_dir, "romanized_informal")

    significance_results: dict = {}
    gap_frames: list[pd.DataFrame] = []

    for name, count_fn in tokenizers.items():
        print(f"  {name} ...", end=" ", flush=True)
        gap_df = per_sentence_gap(
            native_lines,
            romanized_lines,
            count_fn,
            word_count_fn=word_count_fn,
        )
        gap_df.insert(0, "tokenizer", name)
        gap_df.insert(1, "word_count_method", word_count_method)
        gap_frames.append(gap_df)

        try:
            stats = test_gap_significance(
                gap_df, n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            stats["word_count_method"] = word_count_method
            significance_results[name] = stats
            print(
                f"p={stats['p_value']:.4f}  "
                f"median_gap={stats['median_gap']:.3f}  "
                f"CI=[{stats['bootstrap_ci_95_low']:.3f}, {stats['bootstrap_ci_95_high']:.3f}]  "
                f"r_rb={stats['rank_biserial_correlation']:.3f}  "
                f"direction={stats['median_direction']}"
            )
        except ValueError as exc:
            significance_results[name] = {"error": str(exc)}
            print(f"skipped - {exc}")

    sig_path = results_dir / "script_gap_significance.json"
    with sig_path.open("w", encoding="utf-8") as f:
        json.dump(significance_results, f, indent=2)

    if gap_frames:
        pd.concat(gap_frames, ignore_index=True).to_csv(
            results_dir / "per_sentence_gaps.csv", index=False
        )

    print(f"\nWrote {sig_path}")


if __name__ == "__main__":
    main()
