"""Generate romanized_informal by transliterating native_informal to Latin script.

Transliteration (not translation) preserves morphological structure while
producing Latin-script text, creating the line-matched pairs required by
script_gap_stats.per_sentence_gap().  The ITRANS scheme is used because it
is ASCII-safe, reversible, and widely cited in South Asian NLP literature.

Dependency: indic-transliteration (pip install indic-transliteration)
"""

from __future__ import annotations

from pathlib import Path


def collect_romanized(
    native_path: str | Path,
    output_path: str | Path,
) -> int:
    """Transliterate each line of native_path from Telugu to ITRANS romanization.

    Output is line-aligned with native_path — sentence i in romanized_informal
    corresponds to sentence i in native_informal, satisfying the matched-pair
    requirement of script_gap_stats.per_sentence_gap().

    Returns the number of lines written.
    """
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    native_lines = Path(native_path).read_text(encoding="utf-8").splitlines()

    romanized: list[str] = []
    for line in native_lines:
        line = line.strip()
        if not line:
            continue
        romanized.append(transliterate(line, sanscript.TELUGU, sanscript.ITRANS))

    Path(output_path).write_text("\n".join(romanized) + "\n", encoding="utf-8")
    return len(romanized)
