#!/usr/bin/env python3
"""Correlate fertility with published benchmark accuracy scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.analysis.fertility_accuracy_corr import correlate_fertility_accuracy
from telugu_audit.reporting.plots import plot_fertility_accuracy_scatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment folder containing fertility_by_register.csv",
    )
    parser.add_argument(
        "--benchmark-scores",
        required=True,
        help="YAML file mapping model name -> accuracy (human-sourced, cited)",
    )
    parser.add_argument(
        "--register",
        default="native_informal",
        help="Which register's fertility to correlate against",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    fertility_df = pd.read_csv(exp_dir / "results" / "fertility_by_register.csv")
    subset = fertility_df[fertility_df["register"] == args.register]
    fertility_by_model = dict(
        zip(subset["tokenizer"], subset["fertility_tokens_per_word"])
    )

    with Path(args.benchmark_scores).open(encoding="utf-8") as f:
        benchmark_cfg = yaml.safe_load(f)

    accuracy_by_model = benchmark_cfg.get("accuracy_by_model", {})
    if not accuracy_by_model:
        raise ValueError(
            "benchmark-scores YAML must contain accuracy_by_model — "
            "TODO(human): supply cited benchmark numbers"
        )

    result = correlate_fertility_accuracy(fertility_by_model, accuracy_by_model)
    out_path = exp_dir / "results" / "fertility_accuracy_correlation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_fertility_accuracy_scatter(
        fertility_by_model,
        accuracy_by_model,
        exp_dir / "figures",
    )
    print(f"Wrote correlation to {out_path}")


if __name__ == "__main__":
    main()
