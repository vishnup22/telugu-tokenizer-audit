from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from telugu_audit.corpus.schema import (
    DEFAULT_OPTIONAL_REGISTERS,
    DEFAULT_REQUIRED_REGISTERS,
    VALID_MORPH_TYPES,
)

REGISTERS = DEFAULT_REQUIRED_REGISTERS
OPTIONAL_REGISTERS = DEFAULT_OPTIONAL_REGISTERS


def load_corpus(
    corpus_dir: str | Path,
    register: str,
    allowed_registers: tuple[str, ...] | None = None,
) -> list[str]:
    expected = allowed_registers or (REGISTERS + OPTIONAL_REGISTERS)
    if register not in expected:
        raise ValueError(f"Unknown register {register!r}; expected one of {expected}")

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


def load_all_registers(
    corpus_dir: str | Path,
    required_registers: tuple[str, ...] | None = None,
    optional_registers: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    corpus_path = Path(corpus_dir)
    required = required_registers or REGISTERS
    optional = optional_registers or OPTIONAL_REGISTERS
    allowed = tuple(required) + tuple(optional)

    registers = list(required)
    for register in optional:
        if (corpus_path / f"{register}.txt").exists():
            registers.append(register)
    return {
        register: load_corpus(corpus_dir, register, allowed_registers=allowed)
        for register in registers
    }


def load_minimal_pairs(
    path: str | Path,
    valid_morph_types: frozenset[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, comment="#")
    required = {"word", "gloss", "morph_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"minimal_pairs missing columns: {sorted(missing)}")

    allowed_types = valid_morph_types if valid_morph_types is not None else VALID_MORPH_TYPES
    invalid = set(df["morph_type"].dropna()) - allowed_types
    if invalid:
        raise ValueError(
            f"Unknown morph_type values: {sorted(invalid)}. "
            f"Valid types: {sorted(allowed_types)}"
        )

    is_fixture = Path(path).name.startswith("FAKE_")
    if not is_fixture:
        fake_words = [w for w in df["word"].dropna() if "FAKE" in str(w)]
        if fake_words:
            print(
                f"\nERROR: {path} contains placeholder FAKE words: {fake_words[:3]}\n"
                "  Replace with real Telugu minimal pairs before running the pipeline.\n"
                "  See PLAN.md section 7 for what is required.\n",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"Refusing to run: {path} contains FAKE placeholder data. "
                "Supply real minimal pairs or use a FAKE_-prefixed test fixture."
            )

    return df
