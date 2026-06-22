from __future__ import annotations

from pathlib import Path

from telugu_audit.corpus.cleaning import clean_lines
from telugu_audit.corpus.loaders import REGISTERS


def build_corpus(
    raw_dir: Path,
    output_dir: Path,
    registers: tuple[str, ...] = REGISTERS,
) -> dict[str, int]:
    """Clean raw register files and write them to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    line_counts: dict[str, int] = {}

    for register in registers:
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
