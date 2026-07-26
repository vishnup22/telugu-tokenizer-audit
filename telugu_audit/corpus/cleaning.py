from __future__ import annotations

import re

USERNAME_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def strip_pii(text: str) -> str:
    text = USERNAME_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def filter_by_length(
    lines: list[str],
    min_chars: int = 5,
    max_chars: int = 500,
) -> list[str]:
    return [line for line in lines if min_chars <= len(line) <= max_chars]


def clean_lines(lines: list[str]) -> list[str]:
    cleaned = [strip_pii(line) for line in lines]
    cleaned = [line for line in cleaned if line]
    cleaned = deduplicate_lines(cleaned)
    return filter_by_length(cleaned)


def clean_paired_lines(
    native_lines: list[str],
    romanized_lines: list[str],
    min_chars: int = 5,
    max_chars: int = 500,
) -> tuple[list[str], list[str]]:
    """Clean two line-aligned parallel corpora, dropping a pair if either side fails.

    Native/romanized informal registers must stay index-aligned (same sentence,
    two scripts) for per-sentence gap analysis. Cleaning each side independently
    can drop different line indices from each and silently break that pairing.
    """
    if len(native_lines) != len(romanized_lines):
        raise ValueError(
            "Paired corpora must start with equal line counts: "
            f"{len(native_lines)} native vs {len(romanized_lines)} romanized"
        )

    seen_native: set[str] = set()
    seen_romanized: set[str] = set()
    out_native: list[str] = []
    out_romanized: list[str] = []

    for native, romanized in zip(native_lines, romanized_lines):
        native = strip_pii(native)
        romanized = strip_pii(romanized)
        if not native or not romanized:
            continue
        if not (min_chars <= len(native) <= max_chars):
            continue
        if not (min_chars <= len(romanized) <= max_chars):
            continue
        if native in seen_native or romanized in seen_romanized:
            continue
        seen_native.add(native)
        seen_romanized.add(romanized)
        out_native.append(native)
        out_romanized.append(romanized)

    return out_native, out_romanized
