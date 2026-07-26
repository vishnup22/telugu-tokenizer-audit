from __future__ import annotations

from pathlib import Path

from telugu_audit.corpus.cleaning import clean_lines, clean_paired_lines
from telugu_audit.corpus.schema import (
    DEFAULT_OPTIONAL_REGISTERS,
    DEFAULT_REQUIRED_REGISTERS,
    get_language_profiles,
)


DEFAULT_REGISTERS = DEFAULT_REQUIRED_REGISTERS
DEFAULT_OPTIONAL = DEFAULT_OPTIONAL_REGISTERS


def _read_raw(raw_dir: Path, register: str) -> list[str]:
    raw_path = raw_dir / f"{register}.txt"
    if not raw_path.exists():
        fake_path = raw_dir / f"FAKE_{register}.txt"
        raw_path = fake_path if fake_path.exists() else raw_path

    if not raw_path.exists():
        raise FileNotFoundError(f"No raw corpus for {register}: {raw_path}")

    with raw_path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_corpus(
    raw_dir: Path,
    output_dir: Path,
    registers: tuple[str, ...] = DEFAULT_REGISTERS,
    optional_registers: tuple[str, ...] = DEFAULT_OPTIONAL,
    config: dict | None = None,
) -> dict[str, int]:
    """Clean raw register files and write them to output_dir.

    Native/romanized informal registers are cleaned jointly (a line pair is
    dropped from both sides if either fails PII/length/dedup) so downstream
    per-sentence gap analysis keeps its 1:1 alignment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    line_counts: dict[str, int] = {}

    active_registers = list(registers)
    for register in optional_registers:
        if (raw_dir / f"{register}.txt").exists() and register not in active_registers:
            active_registers.append(register)

    active_set = set(active_registers)
    pairs = {
        (profile.native_informal_register, profile.romanized_informal_register)
        for profile in get_language_profiles(config)
    }

    handled: set[str] = set()
    for native_reg, romanized_reg in pairs:
        if native_reg not in active_set or romanized_reg not in active_set:
            continue

        native_raw = _read_raw(raw_dir, native_reg)
        romanized_raw = _read_raw(raw_dir, romanized_reg)
        native_clean, romanized_clean = clean_paired_lines(native_raw, romanized_raw)

        (output_dir / f"{native_reg}.txt").write_text(
            "\n".join(native_clean) + "\n", encoding="utf-8"
        )
        (output_dir / f"{romanized_reg}.txt").write_text(
            "\n".join(romanized_clean) + "\n", encoding="utf-8"
        )
        line_counts[native_reg] = len(native_clean)
        line_counts[romanized_reg] = len(romanized_clean)
        handled.add(native_reg)
        handled.add(romanized_reg)

    for register in active_registers:
        if register in handled:
            continue

        raw_lines = _read_raw(raw_dir, register)
        cleaned = clean_lines(raw_lines)
        out_path = output_dir / f"{register}.txt"
        out_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        line_counts[register] = len(cleaned)

    return line_counts
