from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from telugu_audit.analysis.minimal_pair_stats import (
    flag_high_variance_morph_types,
    kruskal_wallis_by_tokenizer,
    summarize_minimal_pairs,
)


def _csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_summarize_minimal_pairs_column_names(tmp_path):
    csv_path = _csv(
        tmp_path / "mp.csv",
        [
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 2},
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 4},
        ],
    )
    result = summarize_minimal_pairs(csv_path)
    assert set(result.columns) == {
        "tokenizer", "morph_type",
        "mean_tokens", "median_tokens", "n_words", "std_tokens",
        "min_tokens", "max_tokens", "mean_plus_minus_sd",
    }


def test_summarize_minimal_pairs_aggregation_values(tmp_path):
    csv_path = _csv(
        tmp_path / "mp.csv",
        [
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 2},
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 4},
            {"tokenizer": "tok_a", "morph_type": "case_suffix", "n_tokens": 6},
        ],
    )
    result = summarize_minimal_pairs(csv_path)
    bn = result[result["morph_type"] == "base_noun"].iloc[0]
    assert bn["n_words"] == 2
    assert bn["mean_tokens"] == pytest.approx(3.0)
    assert bn["median_tokens"] == pytest.approx(3.0)
    assert bn["std_tokens"] == pytest.approx(math.sqrt(2))
    assert bn["min_tokens"] == 2
    assert bn["max_tokens"] == 4


def test_summarize_minimal_pairs_single_word_std_is_nan(tmp_path):
    csv_path = _csv(
        tmp_path / "mp.csv",
        [{"tokenizer": "tok_a", "morph_type": "case_suffix", "n_tokens": 3}],
    )
    result = summarize_minimal_pairs(csv_path)
    row = result.iloc[0]
    assert row["n_words"] == 1
    assert math.isnan(row["std_tokens"])


def test_flag_high_variance_morph_types_top1(tmp_path):
    csv_path = _csv(
        tmp_path / "mp.csv",
        [
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 2},
            {"tokenizer": "tok_b", "morph_type": "base_noun", "n_tokens": 8},
            {"tokenizer": "tok_a", "morph_type": "case_suffix", "n_tokens": 3},
            {"tokenizer": "tok_b", "morph_type": "case_suffix", "n_tokens": 4},
        ],
    )
    top1 = flag_high_variance_morph_types(csv_path, top_n=1)
    assert top1 == ["base_noun"]


def test_flag_high_variance_morph_types_returns_list(tmp_path):
    csv_path = _csv(
        tmp_path / "mp.csv",
        [
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 5},
            {"tokenizer": "tok_b", "morph_type": "base_noun", "n_tokens": 5},
        ],
    )
    result = flag_high_variance_morph_types(csv_path, top_n=3)
    assert isinstance(result, list)


def test_flag_high_variance_morph_types_ordering(tmp_path):
    csv_path = _csv(
        tmp_path / "mp.csv",
        [
            {"tokenizer": "tok_a", "morph_type": "verb_agglutination_chain", "n_tokens": 1},
            {"tokenizer": "tok_b", "morph_type": "verb_agglutination_chain", "n_tokens": 9},
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 2},
            {"tokenizer": "tok_b", "morph_type": "base_noun", "n_tokens": 6},
            {"tokenizer": "tok_a", "morph_type": "case_suffix", "n_tokens": 3},
            {"tokenizer": "tok_b", "morph_type": "case_suffix", "n_tokens": 4},
        ],
    )
    top2 = flag_high_variance_morph_types(csv_path, top_n=2)
    assert top2[0] == "verb_agglutination_chain"
    assert top2[1] == "base_noun"
    assert len(top2) == 2


def test_kruskal_wallis_by_tokenizer_returns_stats(tmp_path):
    csv_path = _csv(
        tmp_path / "mp.csv",
        [
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 1},
            {"tokenizer": "tok_a", "morph_type": "base_noun", "n_tokens": 2},
            {"tokenizer": "tok_a", "morph_type": "case_suffix", "n_tokens": 7},
            {"tokenizer": "tok_a", "morph_type": "case_suffix", "n_tokens": 8},
        ],
    )
    result = kruskal_wallis_by_tokenizer(csv_path)
    row = result.iloc[0]
    assert row["tokenizer"] == "tok_a"
    assert row["n_categories"] == 2
    assert row["kruskal_h"] >= 0
