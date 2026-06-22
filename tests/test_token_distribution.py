from __future__ import annotations

import pytest

from telugu_audit.analysis.token_distribution import (
    distribution_summary,
    per_line_fertility,
    tokens_per_word_buckets,
)


def _count_fn(text: str) -> int:
    return len(text.split()) * 2  # 2 tokens per word, always


# ---------------------------------------------------------------------------
# per_line_fertility
# ---------------------------------------------------------------------------

def test_per_line_fertility_constant_count_fn():
    lines = ["one two three", "a b"]
    f = per_line_fertility(lines, _count_fn)
    assert len(f) == 2
    assert all(v == pytest.approx(2.0) for v in f)


def test_per_line_fertility_skips_blank_lines():
    lines = ["a b", "", "   ", "c d e"]
    f = per_line_fertility(lines, _count_fn)
    assert len(f) == 2


def test_per_line_fertility_empty_input():
    f = per_line_fertility([], _count_fn)
    assert len(f) == 0


# ---------------------------------------------------------------------------
# distribution_summary
# ---------------------------------------------------------------------------

def test_distribution_summary_keys():
    result = distribution_summary(["a b c"] * 10, _count_fn)
    assert set(result.keys()) == {
        "n_lines", "mean", "std", "p25", "p50", "p75", "p90", "p99"
    }


def test_distribution_summary_constant_fertility():
    result = distribution_summary(["a b"] * 20, _count_fn)
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(0.0)
    assert result["p50"] == pytest.approx(2.0)
    assert result["p99"] == pytest.approx(2.0)
    assert result["n_lines"] == 20


def test_distribution_summary_percentiles_ordered():
    # varied line lengths → varied fertility values
    def varied(text: str) -> int:
        return len(text.split())  # 1 token per word → fertility always 1.0
    lines = [f"{'word ' * i}end" for i in range(1, 21)]
    result = distribution_summary(lines, varied)
    assert result["p25"] <= result["p50"] <= result["p75"] <= result["p90"] <= result["p99"]


# ---------------------------------------------------------------------------
# tokens_per_word_buckets
# ---------------------------------------------------------------------------

def test_tokens_per_word_buckets_all_in_one_bucket():
    words = ["hello"] * 10
    df = tokens_per_word_buckets(words, _count_fn)
    row = df[df["tokens_per_word"] == "2"].iloc[0]
    assert row["n_words"] == 10
    assert row["pct_words"] == pytest.approx(100.0)


def test_tokens_per_word_buckets_pct_sums_to_100():
    def varied(text: str) -> int:
        return {"a": 1, "b": 3, "c": 5}.get(text.strip(), 2)
    words = ["a"] * 4 + ["b"] * 3 + ["c"] * 3
    df = tokens_per_word_buckets(words, varied)
    assert df["pct_words"].sum() == pytest.approx(100.0)


def test_tokens_per_word_buckets_five_plus_bucket():
    def many_tokens(_: str) -> int:
        return 6
    words = ["x"] * 5
    df = tokens_per_word_buckets(words, many_tokens)
    row = df[df["tokens_per_word"] == "5+"].iloc[0]
    assert row["n_words"] == 5


def test_tokens_per_word_buckets_all_buckets_present():
    df = tokens_per_word_buckets(["x"] * 5, _count_fn)
    assert set(df["tokens_per_word"]) == {"1", "2", "3", "4", "5+"}


def test_tokens_per_word_buckets_empty_input():
    df = tokens_per_word_buckets([], _count_fn)
    assert len(df) == 0
