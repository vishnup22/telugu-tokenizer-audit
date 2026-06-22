from __future__ import annotations

import pytest

from telugu_audit.metrics.cost_translation import (
    cost_and_context_for_model,
    cost_per_content,
    effective_context_words,
)

_PRICING = {
    "models": {
        "test-model": {
            "price_per_1k_input_tokens": 2.0,
            "context_limit_tokens": 1000,
        },
        "free-model": {
            "price_per_1k_input_tokens": 0.0,
            "context_limit_tokens": 8192,
        },
    }
}


# ---------------------------------------------------------------------------
# cost_per_content
# ---------------------------------------------------------------------------

def test_cost_per_content_basic():
    assert cost_per_content(1000, 2.0) == pytest.approx(2.0)


def test_cost_per_content_fractional():
    assert cost_per_content(500, 1.0) == pytest.approx(0.5)


def test_cost_per_content_zero_tokens():
    assert cost_per_content(0, 5.0) == pytest.approx(0.0)


def test_cost_per_content_zero_price():
    assert cost_per_content(1000, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# effective_context_words
# ---------------------------------------------------------------------------

def test_effective_context_words_basic():
    assert effective_context_words(1000, 2.0) == pytest.approx(500.0)


def test_effective_context_words_fertility_one():
    assert effective_context_words(8192, 1.0) == pytest.approx(8192.0)


def test_effective_context_words_zero_fertility_returns_zero():
    assert effective_context_words(128000, 0) == pytest.approx(0.0)


def test_effective_context_words_zero_context():
    assert effective_context_words(0, 2.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cost_and_context_for_model
# ---------------------------------------------------------------------------

def test_cost_and_context_for_model_keys():
    result = cost_and_context_for_model(_PRICING, "test-model", 500, 2.0)
    assert set(result.keys()) == {
        "model",
        "cost_usd",
        "effective_context_words",
        "context_limit_tokens",
        "price_per_1k_input_tokens",
    }


def test_cost_and_context_for_model_values():
    result = cost_and_context_for_model(_PRICING, "test-model", 500, 2.0)
    assert result["model"] == "test-model"
    assert result["cost_usd"] == pytest.approx(1.0)          # (500/1000)*2.0
    assert result["effective_context_words"] == pytest.approx(500.0)  # 1000/2.0
    assert result["context_limit_tokens"] == 1000
    assert result["price_per_1k_input_tokens"] == pytest.approx(2.0)


def test_cost_and_context_for_model_zero_price():
    result = cost_and_context_for_model(_PRICING, "free-model", 1000, 1.5)
    assert result["cost_usd"] == pytest.approx(0.0)


def test_cost_and_context_for_model_zero_fertility():
    result = cost_and_context_for_model(_PRICING, "test-model", 500, 0.0)
    assert result["effective_context_words"] == pytest.approx(0.0)


def test_cost_and_context_for_model_missing_model_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        cost_and_context_for_model(_PRICING, "nonexistent", 500, 2.0)
