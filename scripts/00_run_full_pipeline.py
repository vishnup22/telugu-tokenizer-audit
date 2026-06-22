#!/usr/bin/env python3
"""Run the full audit pipeline under a single experiment directory.

Stages (each can also be run standalone via scripts/01–05):
  1. Build processed corpus from raw sources
  2. Fertility audit + script-fairness gap
  3. Minimal-pair audit
  4. Export tables, figures, and summary CSVs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.analysis.fertility_audit import run_fertility_stage
from telugu_audit.analysis.minimal_pair_audit import run_minimal_pair_stage
from telugu_audit.corpus.build import build_corpus
from telugu_audit.reporting.export import run_export_stage
from telugu_audit.run_utils import (
    load_yaml_config,
    new_experiment_dir,
    snapshot_config,
    write_run_metadata,
)
from telugu_audit.tokenizers.registry import get_tokenizer_versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Raw corpus directory for stage 1 (default: data/raw)",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip stage 1; processed corpus must already exist at config corpus_dir",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    run_tag = args.run_tag or config.get("run_tag", "dev")
    config["run_tag"] = run_tag

    exp_dir = new_experiment_dir(run_tag, base_dir=args.experiments_dir)
    snapshot_config(config, exp_dir)
    print(f"Experiment dir: {exp_dir}")

    if not args.skip_corpus:
        raw_dir = Path(args.raw_dir or "data/raw")
        counts = build_corpus(raw_dir, Path(config["corpus_dir"]))
        for reg, n in counts.items():
            print(f"  {reg}: {n} lines")
        print("Stage 1 complete: corpus built")

    corpus_line_counts = run_fertility_stage(exp_dir, config, config_path)
    print("Stage 2 complete: fertility + fairness gap")

    run_minimal_pair_stage(exp_dir, config, config_path)
    print("Stage 3 complete: minimal pair audit")

    run_export_stage(exp_dir)
    print("Stage 4 complete: tables and figures exported")

    write_run_metadata(exp_dir, get_tokenizer_versions(), corpus_line_counts)
    print(f"Done -> {exp_dir}")


if __name__ == "__main__":
    main()
