"""Collect real Tenglish sentences from raw source exports.

This collector is intentionally source-agnostic. It expects you to place raw
exports from YouTube, X, Reddit, or similar public sources into a directory,
then it filters them down to Latin-script Telugu-like comments suitable for a
`tenglish_informal` register.

Supported input formats:
- .jsonl / .json
- .csv / .tsv

Expected text fields (any one is enough):
- text, body, content, comment, selftext, message

Expected metadata fields are preserved when present:
- source, source_id, url, channel, platform
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Iterable

from telugu_audit.corpus.cleaning import deduplicate_lines

_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"[@#]\w+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MULTI_SPACE_RE = re.compile(r"\s+")

# A lightweight set of Telugu-in-Latin cues. The goal is to bias toward
# actual Telugu comments, not generic English code-mixed text.
_TELUGU_CUES = {
    "nenu",
    "neenu",
    "nuvvu",
    "meeru",
    "memu",
    "manam",
    "adi",
    "idhi",
    "ava",
    "avi",
    "ela",
    "enduku",
    "inka",
    "kani",
    "ante",
    "ledu",
    "undhi",
    "undi",
    "unnadu",
    "unnaru",
    "untadi",
    "avunu",
    "bagundi",
    "chala",
    "chaala",
    "manchi",
    "mari",
    "sare",
    "correcte",
    "super",
    "mass",
    "okka",
    "kuda",
    "chesaru",
    "chesthunnaru",
    "chestunnaru",
    "chusanu",
    "chusaru",
    "vastunna",
    "vastundi",
    "vachanu",
    "vachindi",
    "ledu",
    "lekapothe",
    "nuvvu",
    "naaku",
    "naku",
    "mee",
    "maa",
    "meeku",
    "ikuva",
    "kavali",
    "kalalu",
    "cinema",
    "movie",
    "song",
}

_TELUGU_SUFFIX_PATTERNS = (
    r"(anu|avu|adu|indi|unnadu|unnaru|unta|untadi|tunnanu|tunnadu|tunnaru)$",
    r"(ga|ki|lo|ni|tho|nundi|kosam|meeda)$",
)


def _normalize_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("\u200b", " ")
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if "A" <= c <= "Z" or "a" <= c <= "z")
    return latin / len(letters)


def _english_stopword_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "to",
        "of",
        "for",
        "with",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "this",
        "that",
        "it",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
    }
    return sum(1 for tok in tokens if tok.lower() in stopwords) / len(tokens)


def _cue_score(tokens: list[str]) -> int:
    score = 0
    lower_tokens = [tok.lower().strip(".,!?;:()[]{}'\"") for tok in tokens]
    for tok in lower_tokens:
        if tok in _TELUGU_CUES:
            score += 2
        if any(re.search(pattern, tok) for pattern in _TELUGU_SUFFIX_PATTERNS):
            score += 1
    return score


def _looks_like_tenglish(text: str) -> bool:
    text = _normalize_text(text)
    if not text:
        return False
    if _URL_RE.search(text):
        return False
    if len(text.split()) < 5 or len(text.split()) > 50:
        return False

    latin_ratio = _latin_ratio(text)
    if latin_ratio < 0.72:
        return False

    tokens = _LATIN_WORD_RE.findall(text)
    if len(tokens) < 4:
        return False

    if _english_stopword_ratio(tokens) > 0.65:
        return False

    return _cue_score(tokens) >= 2


def _iter_source_records(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
        elif isinstance(payload, dict):
            records = payload.get("data") or payload.get("records") or payload.get("items")
            if isinstance(records, list):
                for item in records:
                    if isinstance(item, dict):
                        yield item
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                yield dict(row)


def _extract_text(record: dict) -> str | None:
    for key in ("text", "body", "content", "comment", "selftext", "message"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _source_label(path: Path, record: dict) -> str:
    for key in ("source", "platform", "site"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    name = path.stem.lower()
    if "youtube" in name:
        return "youtube"
    if "reddit" in name:
        return "reddit"
    if name.startswith("x_") or "twitter" in name or name.startswith("tweet"):
        return "x"
    return path.stem


def collect_tenglish_informal(
    sources_dir: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    n_samples: int = 1000,
    seed: int = 42,
) -> int:
    """Filter source exports into a `tenglish_informal` corpus.

    The output is a curated text file of Latin-script Telugu comments. The
    accompanying metadata file is optional but recommended for provenance.
    """
    source_dir = Path(sources_dir)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Tenglish sources directory not found: {source_dir}"
        )

    source_files = [
        p for p in sorted(source_dir.rglob("*"))
        if p.suffix.lower() in {".jsonl", ".json", ".csv", ".tsv"}
    ]
    if not source_files:
        raise FileNotFoundError(
            f"No supported source files found in {source_dir}. "
            "Expected .jsonl, .json, .csv, or .tsv exports."
        )

    candidates: list[dict] = []
    for path in source_files:
        for record in _iter_source_records(path):
            if not isinstance(record, dict):
                continue
            text = _extract_text(record)
            if not text:
                continue
            normalized = _normalize_text(text)
            if not _looks_like_tenglish(normalized):
                continue
            candidates.append(
                {
                    "text": normalized,
                    "source": _source_label(path, record),
                    "source_file": str(path),
                    "source_id": record.get("id") or record.get("cid") or record.get("comment_id") or record.get("post_id") or "",
                    "url": record.get("url") or record.get("permalink") or record.get("link") or "",
                    "channel": record.get("channel") or record.get("author") or record.get("username") or "",
                }
            )

    unique_texts = deduplicate_lines([row["text"] for row in candidates])
    by_text: dict[str, dict] = {}
    for row in candidates:
        if row["text"] not in by_text:
            by_text[row["text"]] = row

    ordered = [by_text[text] for text in unique_texts]
    if not ordered:
        raise RuntimeError(
            f"No Tenglish candidates survived filtering in {source_dir}. "
            "Relax the filters or provide better source exports."
        )

    rng = random.Random(seed)
    sampled = rng.sample(ordered, min(n_samples, len(ordered)))
    texts = [row["text"] for row in sampled]

    Path(output_path).write_text("\n".join(texts) + "\n", encoding="utf-8")

    if metadata_path is not None:
        meta_path = Path(metadata_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            for row in sampled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(texts)
