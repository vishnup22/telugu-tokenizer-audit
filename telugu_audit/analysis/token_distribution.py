from __future__ import annotations

from pathlib import Path

import pandas as pd

from telugu_audit.metrics.fertility import word_count


def per_line_fertility(lines: list[str], count_fn, word_count_fn=word_count) -> pd.Series:
    """Per-line fertility (tokens / word-count denominator) for a corpus register."""
    result = []
    for line in lines:
        words = word_count_fn(line)
        if words == 0:
            continue
        result.append(count_fn(line) / words)
    return pd.Series(result, name="fertility")


def distribution_summary(lines: list[str], count_fn, word_count_fn=word_count) -> dict:
    """Percentile summary of per-line fertility."""
    f = per_line_fertility(lines, count_fn, word_count_fn=word_count_fn)
    return {
        "n_lines": int(len(f)),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "p25": float(f.quantile(0.25)),
        "p50": float(f.quantile(0.50)),
        "p75": float(f.quantile(0.75)),
        "p90": float(f.quantile(0.90)),
        "p99": float(f.quantile(0.99)),
    }


def tokens_per_word_buckets(words: list[str], count_fn) -> pd.DataFrame:
    """Distribution of token counts per word: 1 / 2 / 3 / 4 / 5+ buckets."""

    def _bucket(n: int) -> str:
        return str(n) if n <= 4 else "5+"

    counts = [count_fn(w) for w in words if w.strip()]
    if not counts:
        return pd.DataFrame(columns=["tokens_per_word", "n_words", "pct_words"])

    series = pd.Series([_bucket(c) for c in counts])
    order = ["1", "2", "3", "4", "5+"]
    vc = series.value_counts().reindex(order, fill_value=0)
    df = vc.reset_index()
    df.columns = ["tokens_per_word", "n_words"]
    total = df["n_words"].sum()
    df["pct_words"] = (df["n_words"] / total * 100).round(1) if total > 0 else 0.0
    return df


def run_distribution_stage(
    experiment_dir: Path,
    corpora: dict[str, list[str]],
    tokenizers: dict,
    word_count_fn=word_count,
    word_count_method: str = "whitespace",
) -> None:
    """Write per-register fertility percentiles and token-length histograms."""
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    dist_rows = []
    bucket_frames = []

    for tok_name, count_fn in tokenizers.items():
        for register, lines in corpora.items():
            summary = distribution_summary(lines, count_fn, word_count_fn=word_count_fn)
            dist_rows.append(
                {
                    "tokenizer": tok_name,
                    "register": register,
                    "word_count_method": word_count_method,
                    **summary,
                }
            )

            words = [w for line in lines for w in line.split() if w]
            buckets = tokens_per_word_buckets(words, count_fn)
            buckets.insert(0, "register", register)
            buckets.insert(0, "tokenizer", tok_name)
            bucket_frames.append(buckets)

    pd.DataFrame(dist_rows).to_csv(
        results_dir / "fertility_distribution.csv", index=False
    )
    pd.concat(bucket_frames, ignore_index=True).to_csv(
        results_dir / "token_length_dist.csv", index=False
    )
