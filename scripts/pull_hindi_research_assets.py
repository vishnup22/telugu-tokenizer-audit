#!/usr/bin/env python3
"""Pull stronger Hindi research assets into data/raw/.

Outputs:
  data/raw/hi_native_formal.txt
  data/raw/hi_native_informal.txt
  data/raw/hi_romanized_informal.txt
  data/raw/hi_organic_romanized_informal.txt
  data/raw/hindi_sources.md

This script upgrades the initial Hindi package by:
  1. topping up formal Hindi from multiple Wikipedia-derived sources
  2. rebuilding informal Hindi from the current social sources
  3. keeping the aligned transliterated romanized file
  4. adding a separate organic Hinglish file from naturally occurring sources

The organic Hinglish file is still a heuristic research input, not a gold
register. It should be reported separately from the aligned transliterated file.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[\"'`“”‘’()\[\]{}<>|]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")

HINDI_MARKERS = {
    "accha",
    "acha",
    "bahut",
    "bhi",
    "hai",
    "hain",
    "ho",
    "hua",
    "kar",
    "karo",
    "kya",
    "kyu",
    "kyun",
    "mai",
    "main",
    "mera",
    "meri",
    "nahi",
    "nhi",
    "nhii",
    "pata",
    "sab",
    "se",
    "tha",
    "thi",
    "toh",
    "tum",
    "yaar",
}

FORMAL_SOURCES = [
    {"dataset": "vibrantlabsai/hindi-wikipedia", "split": "train", "field": "text"},
    {"dataset": "zicsx/Wikipedia-Hindi", "split": "train", "field": "text"},
    {"dataset": "marsh-mellow/hindi_wikipedia", "split": "train", "field": "text"},
]

INFORMAL_SOURCES = [
    {"dataset": "necrosyth/hindi-social-media-hostility-detection", "split": "train", "field": "Post"},
    {"dataset": "sepidmnorozy/Hindi_sentiment", "split": "train", "field": "text"},
]

ORGANIC_ROMANIZED_SOURCES = [
    {"dataset": "Abhishek4896/hindi-english-code-mixed-tweets-sentiment", "split": "train", "field": "tweet"},
    {"dataset": "shae2977/hinglish-youtube-sentiments-dataset", "split": "train", "field": "comment"},
]


def _clean_text(text: str) -> str:
    text = text.replace("\u200c", " ").replace("\u200d", " ").replace("\ufeff", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in SENTENCE_SPLIT_RE.split(paragraph):
            sentence = _clean_text(sentence)
            if sentence:
                parts.append(sentence)
    return parts


def _contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for ch in letters if LATIN_RE.match(ch))
    return latin / len(letters)


def _looks_native_hindi(text: str, min_len: int, max_len: int) -> bool:
    return min_len <= len(text) <= max_len and _contains_devanagari(text)


def _looks_hinglish(text: str) -> bool:
    if not 10 <= len(text) <= 280:
        return False
    if _contains_devanagari(text):
        return False
    if _latin_ratio(text) < 0.75:
        return False
    tokens = [tok.lower() for tok in text.split() if tok]
    if len(tokens) < 3:
        return False
    return any(tok.strip(".,!?") in HINDI_MARKERS for tok in tokens)


def _load_dataset_rows(source: dict) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(source["dataset"], split=source["split"])
    field = source["field"]
    rows: list[str] = []
    for row in ds:
        value = row.get(field)
        if isinstance(value, str):
            rows.append(value)
    return rows


def _collect_native_lines(sources: list[dict], min_len: int, max_len: int) -> tuple[list[str], dict[str, int]]:
    unique: list[str] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()

    for source in sources:
        rows = _load_dataset_rows(source)
        for row in rows:
            for line in _split_sentences(row):
                if not _looks_native_hindi(line, min_len=min_len, max_len=max_len):
                    continue
                if line in seen:
                    continue
                seen.add(line)
                unique.append(line)
                source_counts[source["dataset"]] += 1

    return unique, dict(source_counts)


def _collect_organic_romanized_lines(sources: list[dict]) -> tuple[list[str], dict[str, int]]:
    unique: list[str] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()

    for source in sources:
        rows = _load_dataset_rows(source)
        for row in rows:
            line = _clean_text(row)
            if not _looks_hinglish(line):
                continue
            normalized = line.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(line)
            source_counts[source["dataset"]] += 1

    return unique, dict(source_counts)


def _sample(lines: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    if len(lines) <= n:
        return list(lines)
    return rng.sample(lines, n)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    formal_total: int,
    informal_total: int,
    organic_total: int,
    written_counts: dict[str, int],
    source_counts: dict[str, dict[str, int]],
) -> None:
    lines = [
        "# Hindi Source Manifest",
        "",
        "Generated by `scripts/pull_hindi_research_assets.py`.",
        "",
        "## Outputs",
        f"- `hi_native_formal.txt`: {written_counts['formal']} lines",
        f"- `hi_native_informal.txt`: {written_counts['informal']} lines",
        f"- `hi_romanized_informal.txt`: {written_counts['transliterated']} lines",
        f"- `hi_organic_romanized_informal.txt`: {written_counts['organic']} lines",
        "",
        "## Source Summary",
        f"- Formal native Hindi candidate pool after cleaning/deduplication: {formal_total}",
        f"- Informal native Hindi candidate pool after cleaning/deduplication: {informal_total}",
        f"- Organic romanized Hindi candidate pool after filtering/deduplication: {organic_total}",
        "",
        "## Formal Hindi Sources",
    ]
    for dataset, count in sorted(source_counts["formal"].items()):
        lines.append(f"- `{dataset}`: {count} accepted candidates")

    lines.extend(
        [
            "",
            "## Informal Hindi Sources",
        ]
    )
    for dataset, count in sorted(source_counts["informal"].items()):
        lines.append(f"- `{dataset}`: {count} accepted candidates")

    lines.extend(
        [
            "",
            "## Organic Romanized Hindi Sources",
            "- These are naturally occurring Hinglish/romanized comments or tweets.",
            "- They are not line-aligned with `hi_native_informal.txt`.",
            "- They should be analyzed separately from the transliterated romanized register.",
        ]
    )
    for dataset, count in sorted(source_counts["organic"].items()):
        lines.append(f"- `{dataset}`: {count} accepted candidates")

    lines.extend(
        [
            "",
            "## Notes",
            "- `hi_romanized_informal.txt` is derived by transliterating `hi_native_informal.txt` from Devanagari to ITRANS.",
            "- `hi_organic_romanized_informal.txt` uses heuristic filtering for Latin-dominant Hindi/Hinglish and is suitable as a first-pass organic register, not a gold-standard corpus.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-formal", type=int, default=1000)
    parser.add_argument("--n-informal", type=int, default=1000)
    parser.add_argument("--n-organic", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_candidates, formal_source_counts = _collect_native_lines(
        FORMAL_SOURCES,
        min_len=30,
        max_len=400,
    )
    informal_candidates, informal_source_counts = _collect_native_lines(
        INFORMAL_SOURCES,
        min_len=10,
        max_len=500,
    )
    organic_candidates, organic_source_counts = _collect_organic_romanized_lines(
        ORGANIC_ROMANIZED_SOURCES
    )

    formal_lines = _sample(formal_candidates, args.n_formal, args.seed)
    informal_lines = _sample(informal_candidates, args.n_informal, args.seed)
    organic_lines = _sample(organic_candidates, args.n_organic, args.seed)
    transliterated_lines = [
        transliterate(line, sanscript.DEVANAGARI, sanscript.ITRANS)
        for line in informal_lines
    ]

    _write_lines(output_dir / "hi_native_formal.txt", formal_lines)
    _write_lines(output_dir / "hi_native_informal.txt", informal_lines)
    _write_lines(output_dir / "hi_romanized_informal.txt", transliterated_lines)
    _write_lines(output_dir / "hi_organic_romanized_informal.txt", organic_lines)

    _write_manifest(
        output_dir / "hindi_sources.md",
        formal_total=len(formal_candidates),
        informal_total=len(informal_candidates),
        organic_total=len(organic_candidates),
        written_counts={
            "formal": len(formal_lines),
            "informal": len(informal_lines),
            "transliterated": len(transliterated_lines),
            "organic": len(organic_lines),
        },
        source_counts={
            "formal": formal_source_counts,
            "informal": informal_source_counts,
            "organic": organic_source_counts,
        },
    )

    print(f"Formal candidates: {len(formal_candidates)} -> wrote {len(formal_lines)}")
    print(f"Informal candidates: {len(informal_candidates)} -> wrote {len(informal_lines)}")
    print(f"Organic romanized candidates: {len(organic_candidates)} -> wrote {len(organic_lines)}")
    print(f"Wrote manifest: {output_dir / 'hindi_sources.md'}")


if __name__ == "__main__":
    main()
