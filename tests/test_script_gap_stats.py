from __future__ import annotations

import pandas as pd
import pytest

from telugu_audit.analysis.script_gap_stats import (
    per_sentence_gap,
    rank_biserial_correlation,
    test_gap_significance as check_gap_significance,
)


def _count_fn(text: str) -> int:
    return sum(3 if w[0].isupper() else 2 for w in text.split() if w)


_NATIVE = ["Telugu Telugu Telugu"] * 20
_ROMAN = ["roman roman roman"] * 20


def test_per_sentence_gap_row_count():
    df = per_sentence_gap(_NATIVE, _ROMAN, _count_fn)
    assert len(df) == 20


def test_per_sentence_gap_columns():
    df = per_sentence_gap(_NATIVE, _ROMAN, _count_fn)
    assert set(df.columns) >= {
        "sentence_idx", "native_fertility", "romanized_fertility",
        "fertility_gap", "fertility_ratio",
    }


def test_per_sentence_gap_values():
    df = per_sentence_gap(_NATIVE, _ROMAN, _count_fn)
    assert df["native_fertility"].tolist() == pytest.approx([3.0] * 20)
    assert df["romanized_fertility"].tolist() == pytest.approx([2.0] * 20)
    assert df["fertility_gap"].tolist() == pytest.approx([1.0] * 20)
    assert df["fertility_ratio"].tolist() == pytest.approx([1.5] * 20)


def test_per_sentence_gap_unequal_lengths_raises():
    with pytest.raises(ValueError, match="Line counts differ"):
        per_sentence_gap(["a"] * 3, ["b"] * 5, _count_fn)


def test_per_sentence_gap_skips_blank_lines():
    native = ["Telugu Telugu", "", "Telugu Telugu Telugu"]
    roman = ["roman roman", "", "roman roman roman"]
    df = per_sentence_gap(native, roman, _count_fn)
    assert len(df) == 2


def test_significance_detects_positive_gap():
    gap_df = pd.DataFrame({"fertility_gap": [1.0] * 20})
    result = check_gap_significance(gap_df, n_bootstrap=500)
    assert result["p_value"] < 0.05
    assert result["pct_pairs_native_costlier"] == pytest.approx(100.0)
    assert result["n_pairs"] == 20


def test_significance_ci_bounds_ordered():
    gap_df = pd.DataFrame({"fertility_gap": [0.5 + i * 0.01 for i in range(20)]})
    result = check_gap_significance(gap_df, n_bootstrap=500)
    assert result["bootstrap_ci_95_low"] <= result["bootstrap_ci_95_high"]
    assert result["bootstrap_ci_95_low"] > 0


def test_significance_result_keys():
    gap_df = pd.DataFrame({"fertility_gap": [0.5] * 15})
    result = check_gap_significance(gap_df, n_bootstrap=200)
    assert set(result.keys()) == {
        "n_pairs", "n_nonzero_pairs", "median_gap", "mean_gap", "median_direction",
        "pct_pairs_native_costlier", "wilcoxon_statistic", "p_value",
        "rank_biserial_correlation", "bootstrap_ci_95_low", "bootstrap_ci_95_high",
    }


def test_significance_too_few_pairs_raises():
    gap_df = pd.DataFrame({"fertility_gap": [0.5] * 5})
    with pytest.raises(ValueError, match="Too few"):
        check_gap_significance(gap_df)


def test_significance_seed_reproducible():
    gap_df = pd.DataFrame({"fertility_gap": [0.3 + i * 0.05 for i in range(15)]})
    r1 = check_gap_significance(gap_df, n_bootstrap=200, seed=7)
    r2 = check_gap_significance(gap_df, n_bootstrap=200, seed=7)
    assert r1["bootstrap_ci_95_low"] == r2["bootstrap_ci_95_low"]


def test_rank_biserial_correlation_handles_all_positive_deltas():
    assert rank_biserial_correlation(pd.Series([1.0] * 10)) == pytest.approx(1.0)


def test_significance_reports_direction_and_effect_size():
    gap_df = pd.DataFrame({"fertility_gap": [-0.5] * 20})
    result = check_gap_significance(gap_df, n_bootstrap=200)
    assert result["median_direction"] == "romanized_costlier"
    assert result["rank_biserial_correlation"] < 0
