#!/usr/bin/env python3
"""Export tables and figures from an experiment run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.reporting.export import run_export_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment folder to export from",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    run_export_stage(exp_dir)
    print(f"Exported tables and figures under {exp_dir}")


if __name__ == "__main__":
    main()
