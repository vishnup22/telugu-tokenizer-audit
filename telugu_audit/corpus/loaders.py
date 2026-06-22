from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REGISTERS = ("native_formal", "native_informal", "romanized_informal")

VALID_MORPH_TYPES = frozenset(
    {
        "base_noun",
        "case_suffix",
        "case_suffix_with_sandhi",
        "plural_suffix",
        "honorific_suffix",
        "compound_with_sandhi",
        "borrowed_base",
        "borrowed_plus_case_suffix",
        "verb_agglutination_chain",
    }
)


def load_corpus(corpus_dir: str | Path, register: str) -> list[str]:
    if register not in REGISTERS:
        raise ValueError(f"Unknown register {register!r}; expected one of {REGISTERS}")

    path = Path(corpus_dir) / f"{register}.txt"
    if not path.exists():
        raise FileNotFoundError(path)

    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def load_all_registers(corpus_dir: str | Path) -> dict[str, list[str]]:
    return {register: load_corpus(corpus_dir, register) for register in REGISTERS}


def load_minimal_pairs(path: str | Path) -> pd.DataFrame:
    # lines starting with # are human-readable comments (e.g. TODO notes)
    df = pd.read_csv(path, sep="\t", dtype=str, comment="#")
    required = {"word", "gloss", "morph_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"minimal_pairs missing columns: {sorted(missing)}")

    invalid = set(df["morph_type"].dropna()) - VALID_MORPH_TYPES
    if invalid:
        raise ValueError(
            f"Unknown morph_type values: {sorted(invalid)}. "
            f"Valid types: {sorted(VALID_MORPH_TYPES)}"
        )

    # Test fixtures are named FAKE_*.tsv by convention; skip the check for them.
    is_fixture = Path(path).name.startswith("FAKE_")
    if not is_fixture:
        fake_words = [w for w in df["word"].dropna() if "FAKE" in str(w)]
        if fake_words:
            print(
                f"\nERROR: {path} contains placeholder FAKE words: {fake_words[:3]}\n"
                "  Replace with real Telugu minimal pairs before running the pipeline.\n"
                "  See PLAN.md §7 for what is required.\n",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"Refusing to run: {path} contains FAKE placeholder data. "
                "Supply real minimal pairs or use a FAKE_-prefixed test fixture."
            )

    return df
