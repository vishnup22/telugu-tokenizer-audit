from __future__ import annotations


def parity_ratio(telugu_fertility: float, english_fertility: float) -> float:
    if english_fertility == 0:
        raise ValueError("english_fertility must be non-zero")
    return telugu_fertility / english_fertility


def compute_parity_from_lines(
    count_fn,
    telugu_lines: list[str],
    english_lines: list[str],
) -> float:
    from telugu_audit.metrics.fertility import compute_fertility

    if len(telugu_lines) != len(english_lines):
        raise ValueError("Telugu and English line lists must be the same length")

    te = compute_fertility(count_fn, telugu_lines)
    en = compute_fertility(count_fn, english_lines)
    return parity_ratio(
        te["fertility_tokens_per_word"],
        en["fertility_tokens_per_word"],
    )
