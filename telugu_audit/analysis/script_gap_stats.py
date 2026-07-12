from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

from telugu_audit.metrics.fertility import word_count


def per_sentence_gap(
    native_lines: list[str],
    romanized_lines: list[str],
    count_fn,
    word_count_fn=word_count,
) -> pd.DataFrame:
    """Per-sentence fertility and gap for matched native/romanized pairs."""
    if len(native_lines) != len(romanized_lines):
        raise ValueError(
            f"Line counts differ: {len(native_lines)} native vs "
            f"{len(romanized_lines)} romanized - corpus must be matched pairs"
        )
    rows = []
    for i, (nat, rom) in enumerate(zip(native_lines, romanized_lines)):
        nat_words = word_count_fn(nat)
        rom_words = word_count_fn(rom)
        if nat_words == 0 or rom_words == 0:
            continue
        nat_f = count_fn(nat) / nat_words
        rom_f = count_fn(rom) / rom_words
        rows.append(
            {
                "sentence_idx": i,
                "native_fertility": nat_f,
                "romanized_fertility": rom_f,
                "fertility_gap": nat_f - rom_f,
                "fertility_ratio": nat_f / rom_f if rom_f > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def per_sentence_fertility(
    lines: list[str],
    count_fn,
    word_count_fn=word_count,
    register: str | None = None,
) -> pd.DataFrame:
    rows = []
    for i, line in enumerate(lines):
        n_words = word_count_fn(line)
        if n_words == 0:
            continue
        rows.append(
            {
                "sentence_idx": i,
                "register": register,
                "fertility": count_fn(line) / n_words,
            }
        )
    return pd.DataFrame(rows)


def rank_biserial_correlation(differences: pd.Series | np.ndarray) -> float:
    diffs = np.asarray(pd.Series(differences).dropna(), dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0

    ranks = rankdata(np.abs(diffs), method="average")
    positive_rank_sum = float(ranks[diffs > 0].sum())
    negative_rank_sum = float(ranks[diffs < 0].sum())
    total_rank_sum = positive_rank_sum + negative_rank_sum
    if total_rank_sum == 0:
        return 0.0
    return (positive_rank_sum - negative_rank_sum) / total_rank_sum


def bootstrap_median_ci(
    values: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    sample = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(sample) == 0:
        return float("nan"), float("nan")
    if np.all(sample == sample[0]):
        value = float(sample[0])
        return value, value

    try:
        rng = np.random.default_rng(seed)
        result = stats.bootstrap(
            (sample,),
            np.median,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="BCa",
            random_state=rng,
        )
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if np.isnan(low) or np.isnan(high):
            raise ValueError("degenerate BCa interval")
        return (low, high)
    except Exception:
        rng = np.random.default_rng(seed)
        boot_medians = np.array(
            [
                np.median(rng.choice(sample, size=len(sample), replace=True))
                for _ in range(n_resamples)
            ]
        )
        alpha = (1 - confidence_level) / 2
        return (
            float(np.percentile(boot_medians, alpha * 100)),
            float(np.percentile(boot_medians, (1 - alpha) * 100)),
        )


def bootstrap_median_difference_ci(
    native_values: pd.Series | np.ndarray,
    other_values: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    native = np.asarray(pd.Series(native_values).dropna(), dtype=float)
    other = np.asarray(pd.Series(other_values).dropna(), dtype=float)
    if len(native) == 0 or len(other) == 0:
        return float("nan"), float("nan")

    try:
        rng = np.random.default_rng(seed)
        boot_diffs = np.array(
            [
                np.median(rng.choice(native, size=len(native), replace=True))
                - np.median(rng.choice(other, size=len(other), replace=True))
                for _ in range(n_resamples)
            ]
        )
        alpha = (1 - confidence_level) / 2
        return (
            float(np.percentile(boot_diffs, alpha * 100)),
            float(np.percentile(boot_diffs, (1 - alpha) * 100)),
        )
    except Exception:
        return float("nan"), float("nan")


def test_gap_significance(
    gap_df: pd.DataFrame,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Two-sided Wilcoxon test and BCa bootstrap CI on the median fertility gap."""
    gaps = gap_df["fertility_gap"].dropna()
    if len(gaps) < 10:
        raise ValueError(
            f"Too few sentence pairs for a reliable test "
            f"(n={len(gaps)}, need >= 10)"
        )

    stat, p_value = stats.wilcoxon(
        gaps,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    effect_cl = float((gaps > 0).mean())
    ci_low, ci_high = bootstrap_median_ci(gaps, n_resamples=n_bootstrap, seed=seed)
    median_gap = float(gaps.median())
    median_direction = (
        "native_costlier"
        if median_gap > 0
        else "romanized_costlier" if median_gap < 0 else "neutral"
    )
    nonzero_gaps = gaps[gaps != 0]

    return {
        "n_pairs": int(len(gaps)),
        "n_nonzero_pairs": int(len(nonzero_gaps)),
        "median_gap": median_gap,
        "mean_gap": float(gaps.mean()),
        "median_direction": median_direction,
        "pct_pairs_native_costlier": round(effect_cl * 100, 1),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
        "rank_biserial_correlation": float(rank_biserial_correlation(gaps)),
        "bootstrap_ci_95_low": ci_low,
        "bootstrap_ci_95_high": ci_high,
    }


def test_unpaired_gap_significance(
    native_df: pd.DataFrame,
    other_df: pd.DataFrame,
    native_col: str = "fertility",
    other_col: str = "fertility",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Two-sided Mann-Whitney test for an unpaired Tenglish-style comparison."""
    native = pd.Series(native_df[native_col]).dropna().astype(float)
    other = pd.Series(other_df[other_col]).dropna().astype(float)
    if len(native) < 10 or len(other) < 10:
        raise ValueError(
            f"Too few observations for an unpaired test (native={len(native)}, other={len(other)})"
        )

    stat, p_value = stats.mannwhitneyu(native, other, alternative="two-sided")
    median_gap = float(native.median() - other.median())
    median_direction = (
        "native_costlier"
        if median_gap > 0
        else "other_costlier" if median_gap < 0 else "neutral"
    )
    ci_low, ci_high = bootstrap_median_difference_ci(
        native, other, n_resamples=n_bootstrap, seed=seed
    )
    rank_biserial = float((2 * stat) / (len(native) * len(other)) - 1)

    return {
        "n_native": int(len(native)),
        "n_other": int(len(other)),
        "median_gap": median_gap,
        "mean_gap": float(native.mean() - other.mean()),
        "median_direction": median_direction,
        "mannwhitneyu_statistic": float(stat),
        "p_value": float(p_value),
        "rank_biserial_correlation": rank_biserial,
        "bootstrap_ci_95_low": ci_low,
        "bootstrap_ci_95_high": ci_high,
    }
