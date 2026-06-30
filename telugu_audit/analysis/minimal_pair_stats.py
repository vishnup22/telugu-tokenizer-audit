from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import kruskal


def summarize_minimal_pairs(results_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    grouped = (
        df.groupby(["tokenizer", "morph_type"])["n_tokens"]
        .agg(["mean", "median", "count", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "mean_tokens",
            "median": "median_tokens",
            "count": "n_words",
            "std": "std_tokens",
            "min": "min_tokens",
            "max": "max_tokens",
        })
    )
    grouped["mean_plus_minus_sd"] = grouped.apply(
        lambda row: f"{row['mean_tokens']:.2f} +/- {row['std_tokens']:.2f}"
        if pd.notna(row["std_tokens"])
        else f"{row['mean_tokens']:.2f} +/- NA",
        axis=1,
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


def kruskal_wallis_by_tokenizer(results_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    rows: list[dict[str, float | str | int]] = []

    for tokenizer, tok_df in df.groupby("tokenizer"):
        groups = [
            group["n_tokens"].to_numpy()
            for _, group in tok_df.groupby("morph_type")
            if len(group) > 0
        ]
        if len(groups) < 2:
            rows.append(
                {
                    "tokenizer": tokenizer,
                    "n_categories": len(groups),
                    "kruskal_h": float("nan"),
                    "p_value": float("nan"),
                }
            )
            continue

        stat, p_value = kruskal(*groups)
        rows.append(
            {
                "tokenizer": tokenizer,
                "n_categories": len(groups),
                "kruskal_h": float(stat),
                "p_value": float(p_value),
            }
        )

    return pd.DataFrame(rows).sort_values("tokenizer").reset_index(drop=True)
