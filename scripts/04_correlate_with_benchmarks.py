                      
"""Correlate fertility with published benchmark accuracy scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.analysis.posthoc import run_fertility_accuracy_correlation_stage


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

    out_path = run_fertility_accuracy_correlation_stage(
        args.experiment_dir,
        args.benchmark_scores,
        register=args.register,
    )
    print(f"Wrote correlation to {out_path}")


if __name__ == "__main__":
    main()
