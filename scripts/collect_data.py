#!/usr/bin/env python3
"""Collect raw Telugu corpus from Hugging Face and write to data/raw/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.corpus.collectors.english_wikipedia_collector import (
    collect_english_wikipedia,
)
from telugu_audit.corpus.collectors.social_media_collector import collect_social_media
from telugu_audit.corpus.collectors.transliteration_collector import collect_romanized
from telugu_audit.corpus.collectors.tenglish_collector import collect_tenglish_informal
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
    parser.add_argument(
        "--include-english",
        action="store_true",
        help="Also collect english_wiki.txt for the direct English baseline",
    )
    parser.add_argument(
        "--include-tenglish",
        action="store_true",
        help="Also collect tenglish_informal.txt from raw public-source exports",
    )
    parser.add_argument(
        "--tenglish-sources-dir",
        default="data/raw/tenglish_sources",
        help="Directory containing exported YouTube/X/Reddit comments for tenglish_informal",
    )
    parser.add_argument(
        "--tenglish-metadata-output",
        default=None,
        help="Optional metadata JSONL path for accepted Tenglish lines",
    )
    args = parser.parse_args()

    raw_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] native_formal - sampling {args.n_samples} Wikipedia sentences ...")
    n = collect_wikipedia(
        raw_dir / "native_formal.txt",
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"      {n} lines -> {raw_dir / 'native_formal.txt'}")

    print(f"[2/4] native_informal - sampling {args.n_samples} social-media sentences ...")
    n = collect_social_media(
        raw_dir / "native_informal.txt",
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"      {n} lines -> {raw_dir / 'native_informal.txt'}")

    print("[3/4] romanized_informal - transliterating native_informal (ITRANS) ...")
    n = collect_romanized(
        raw_dir / "native_informal.txt",
        raw_dir / "romanized_informal.txt",
    )
    print(f"      {n} lines -> {raw_dir / 'romanized_informal.txt'}")

    if args.include_tenglish:
        sources_dir = Path(args.tenglish_sources_dir)
        metadata_path = (
            Path(args.tenglish_metadata_output)
            if args.tenglish_metadata_output
            else raw_dir / "tenglish_informal.meta.jsonl"
        )
        print(
            f"[4/4] tenglish_informal - filtering public-source exports from {sources_dir} ..."
        )
        n = collect_tenglish_informal(
            sources_dir=sources_dir,
            output_path=raw_dir / "tenglish_informal.txt",
            metadata_path=metadata_path,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        print(f"      {n} lines -> {raw_dir / 'tenglish_informal.txt'}")
    elif args.include_english:
        print(f"[4/4] english_wiki - sampling {args.n_samples} English Wikipedia sentences ...")
        n = collect_english_wikipedia(
            raw_dir / "english_wiki.txt",
            n_samples=args.n_samples,
            seed=args.seed,
        )
        print(f"      {n} lines -> {raw_dir / 'english_wiki.txt'}")

    print(f"\nDone. Raw corpus at {raw_dir}/")
    print("Next: python scripts/01_build_corpus.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
