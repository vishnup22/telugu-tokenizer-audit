from __future__ import annotations

from pathlib import Path

from telugu_audit.analysis.minimal_pair_stats import (
    kruskal_wallis_by_tokenizer,
    summarize_minimal_pairs,
)
from telugu_audit.reporting.plots import export_all_figures
from telugu_audit.reporting.tables import export_tables


def run_export_stage(experiment_dir: Path) -> None:
    """Export tables, figures, and minimal-pair summary artifacts."""
    export_tables(experiment_dir)
    export_all_figures(experiment_dir)

    mp_path = experiment_dir / "results" / "minimal_pair_fertility.csv"
    if mp_path.exists():
        summary = summarize_minimal_pairs(mp_path)
        summary.to_csv(
            experiment_dir / "results" / "minimal_pair_summary.csv", index=False
        )
        kruskal_wallis_by_tokenizer(mp_path).to_csv(
            experiment_dir / "results" / "minimal_pair_kruskal_wallis.csv",
            index=False,
        )
