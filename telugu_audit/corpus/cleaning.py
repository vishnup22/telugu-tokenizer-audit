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
