                      
"""Build processed corpus from raw/interim sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.corpus.build import build_corpus
from telugu_audit.corpus.schema import get_optional_registers, get_required_registers
from telugu_audit.run_utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    raw_dir = Path(args.raw_dir or "data/raw")
    output_dir = Path(args.output_dir or config["corpus_dir"])

    counts = build_corpus(
        raw_dir,
        output_dir,
        registers=get_required_registers(config),
        optional_registers=get_optional_registers(config),
        config=config,
    )
    for register, n in counts.items():
        print(f"{register}: {n} lines -> {output_dir / f'{register}.txt'}")


if __name__ == "__main__":
    main()
