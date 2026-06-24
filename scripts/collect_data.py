#!/usr/bin/env python3
"""Collect raw Telugu corpus from Hugging Face and write to data/raw/.

Stages:
  1. native_formal.txt       -- Telugu Wikipedia sentences
                                (vengi-ai/telugu-wikipedia-clean, CC-BY-SA-4.0)
  2. native_informal.txt     -- social-media sentences
                                (mounikaiiith/Telugu_Sentiment +
                                 mounikaiiith/Telugu-Hatespeech, CC-BY-4.0)
  3. romanized_informal.txt  -- ITRANS transliteration of native_informal
                                (line-matched; required by script_gap_stats)

Run this before scripts/01_build_corpus.py:

  python scripts/collect_data.py --n-samples 1000
  python scripts/01_build_corpus.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.corpus.collectors.social_media_collector import collect_social_media
from telugu_audit.corpus.collectors.transliteration_collector import collect_romanized
from telugu_audit.corpus.collectors.wikipedia_collector import collect_wikipedia


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for raw corpus files (default: data/raw)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Lines to sample per native register (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] native_formal — sampling {args.n_samples} Wikipedia sentences ...")
    n = collect_wikipedia(
        raw_dir / "native_formal.txt",
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"      {n} lines -> {raw_dir / 'native_formal.txt'}")

    print(f"[2/3] native_informal — sampling {args.n_samples} social-media sentences ...")
    n = collect_social_media(
        raw_dir / "native_informal.txt",
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"      {n} lines -> {raw_dir / 'native_informal.txt'}")

    print("[3/3] romanized_informal — transliterating native_informal (ITRANS) ...")
    n = collect_romanized(
        raw_dir / "native_informal.txt",
        raw_dir / "romanized_informal.txt",
    )
    print(f"      {n} lines -> {raw_dir / 'romanized_informal.txt'}")

    print(f"\nDone. Raw corpus at {raw_dir}/")
    print("Next: python scripts/01_build_corpus.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
