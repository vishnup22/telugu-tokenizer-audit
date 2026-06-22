from __future__ import annotations

from pathlib import Path

import pandas as pd


def csv_to_markdown(csv_path: str | Path, max_rows: int | None = None) -> str:
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([headers, separator, *rows])


def csv_to_latex(csv_path: str | Path, max_rows: int | None = None) -> str:
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)
    cols = "l" * len(df.columns)
    lines = [
        "\\begin{tabular}{" + cols + "}",
        "\\toprule",
        " & ".join(df.columns) + " \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def export_tables(experiment_dir: str | Path) -> None:
    results_dir = Path(experiment_dir) / "results"
    tables_dir = Path(experiment_dir) / "tables"
    tables_dir.mkdir(exist_ok=True)

    for csv_path in sorted(results_dir.glob("*.csv")):
        stem = csv_path.stem
        md_path = tables_dir / f"{stem}.md"
        tex_path = tables_dir / f"{stem}.tex"
        md_path.write_text(csv_to_markdown(csv_path), encoding="utf-8")
        tex_path.write_text(csv_to_latex(csv_path), encoding="utf-8")
