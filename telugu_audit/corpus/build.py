from __future__ import annotations

from pathlib import Path

from telugu_audit.corpus.cleaning import clean_lines
from telugu_audit.corpus.schema import DEFAULT_OPTIONAL_REGISTERS, DEFAULT_REQUIRED_REGISTERS


DEFAULT_REGISTERS = DEFAULT_REQUIRED_REGISTERS
DEFAULT_OPTIONAL = DEFAULT_OPTIONAL_REGISTERS


def build_corpus(
    raw_dir: Path,
    output_dir: Path,
    registers: tuple[str, ...] = DEFAULT_REGISTERS,
    optional_registers: tuple[str, ...] = DEFAULT_OPTIONAL,
) -> dict[str, int]:
    """Clean raw register files and write them to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    line_counts: dict[str, int] = {}

    active_registers = list(registers)
    for register in optional_registers:
        if (raw_dir / f"{register}.txt").exists() and register not in active_registers:
            active_registers.append(register)

    for register in active_registers:
        raw_path = raw_dir / f"{register}.txt"
        if not raw_path.exists():
            fake_path = raw_dir / f"FAKE_{register}.txt"
            raw_path = fake_path if fake_path.exists() else raw_path

        if not raw_path.exists():
            raise FileNotFoundError(f"No raw corpus for {register}: {raw_path}")

        with raw_path.open(encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        cleaned = clean_lines(raw_lines)
        out_path = output_dir / f"{register}.txt"
        out_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        line_counts[register] = len(cleaned)

    return line_counts
