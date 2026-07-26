from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import kruskal


def _group_keys(df: pd.DataFrame, *keys: str) -> list[str]:
    """Prepend a 'language' groupby key when the dataframe has one (multilingual runs)."""
    return (["language"] if "language" in df.columns else []) + list(keys)


def summarize_minimal_pairs(results_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    grouped = (
        df.groupby(_group_keys(df, "tokenizer", "morph_type"))["n_tokens"]
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
    index_keys = _group_keys(df, "morph_type")
    pivot = df.pivot_table(
        index=index_keys,
        columns="tokenizer",
        values="n_tokens",
        aggfunc="mean",
    )
    variance = pivot.var(axis=1, skipna=True).reset_index()
    variance.columns = [*index_keys, "tokenizer_variance"]
    return variance.sort_values("tokenizer_variance", ascending=False)


def flag_high_variance_morph_types(
    results_path: str | Path,
    top_n: int = 3,
) -> list[str]:
    ranked = morph_type_variance(results_path)
    return ranked.head(top_n)["morph_type"].tolist()


def kruskal_wallis_by_tokenizer(results_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    has_language = "language" in df.columns
    group_keys = _group_keys(df, "tokenizer")
    rows: list[dict[str, float | str | int]] = []

    for key, tok_df in df.groupby(group_keys):
        key_values = key if isinstance(key, tuple) else (key,)
        groups = [
            group["n_tokens"].to_numpy()
            for _, group in tok_df.groupby("morph_type")
            if len(group) > 0
        ]

        row = dict(zip(group_keys, key_values))
        if len(groups) < 2:
            row.update(
                {
                    "n_categories": len(groups),
                    "kruskal_h": float("nan"),
                    "p_value": float("nan"),
                }
            )
        else:
            stat, p_value = kruskal(*groups)
            row.update(
                {
                    "n_categories": len(groups),
                    "kruskal_h": float(stat),
                    "p_value": float(p_value),
                }
            )
        rows.append(row)

    sort_keys = (["language", "tokenizer"] if has_language else ["tokenizer"])
    return pd.DataFrame(rows).sort_values(sort_keys).reset_index(drop=True)
