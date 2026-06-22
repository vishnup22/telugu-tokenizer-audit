#!/usr/bin/env python3
"""Run fertility audit across registers and compute script-fairness gap.

Pass --experiment-dir to write into an existing folder (e.g. when chaining
with script 03 manually).  Omit it to create a fresh dated experiment folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.analysis.fertility_audit import run_fertility_stage
from telugu_audit.run_utils import (
    load_yaml_config,
    new_experiment_dir,
    snapshot_config,
    write_run_metadata,
)
from telugu_audit.tokenizers.registry import get_tokenizer_versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Existing experiment dir to write into (skips dir creation and metadata write)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        run_fertility_stage(exp_dir, config, config_path)
    else:
        run_tag = args.run_tag or config.get("run_tag", "dev")
        config["run_tag"] = run_tag
        exp_dir = new_experiment_dir(run_tag, base_dir=args.experiments_dir)
        snapshot_config(config, exp_dir)
        corpus_line_counts = run_fertility_stage(exp_dir, config, config_path)
        write_run_metadata(exp_dir, get_tokenizer_versions(), corpus_line_counts)

    print(f"Wrote results to {exp_dir}")


if __name__ == "__main__":
    main()
