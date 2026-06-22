from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_minimal_pairs(results_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    grouped = (
        df.groupby(["tokenizer", "morph_type"])["n_tokens"]
        .agg(["mean", "median", "count", "std"])
        .reset_index()
        .rename(columns={
            "mean": "mean_tokens",
            "median": "median_tokens",
            "count": "n_words",
            "std": "std_tokens",
        })
    )
    return grouped


def morph_type_variance(results_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    pivot = df.pivot_table(
        index="morph_type",
        columns="tokenizer",
        values="n_tokens",
        aggfunc="mean",
    )
    variance = pivot.var(axis=1, skipna=True).reset_index()
    variance.columns = ["morph_type", "tokenizer_variance"]
    return variance.sort_values("tokenizer_variance", ascending=False)


def flag_high_variance_morph_types(
    results_path: str | Path,
    top_n: int = 3,
) -> list[str]:
    ranked = morph_type_variance(results_path)
    return ranked.head(top_n)["morph_type"].tolist()
