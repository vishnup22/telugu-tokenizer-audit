from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

CountFn = Callable[[str], int]


def word_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def indicnlp_word_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    from indicnlp.tokenize import indic_tokenize

    return len(indic_tokenize.trivial_tokenize(stripped, lang="te"))


def get_word_count_fn(method: str) -> CountFn:
    normalized = method.strip().lower()
    if normalized == "whitespace":
        return word_count
    if normalized == "indicnlp":
        from indicnlp.tokenize import indic_tokenize as _indic_tokenize              
        return indicnlp_word_count
    raise ValueError(
        f"Unknown word-count method {method!r}; expected 'whitespace' or 'indicnlp'"
    )


def byte_count(text: str) -> int:
    return len(text.encode("utf-8"))


def compute_fertility(
    count_fn: CountFn,
    lines: list[str],
    word_count_fn: CountFn = word_count,
    word_count_method: str = "whitespace",
) -> dict:
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    n_tokenized = 0

    for line in lines:
        try:
            n = count_fn(line)
        except Exception as exc:
            logger.warning("count_fn failed, skipping line: %s", exc)
            continue
        n_tokenized += 1
        total_tokens += n
        total_words += word_count_fn(line)
        total_bytes += byte_count(line)

    fertility = total_tokens / total_words if total_words else 0.0
    compression = total_bytes / total_tokens if total_tokens else 0.0

    return {
        "n_lines_attempted": len(lines),
        "n_lines_tokenized": n_tokenized,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "fertility_tokens_per_word": fertility,
        "compression_bytes_per_token": compression,
        "word_count_method": word_count_method,
    }
