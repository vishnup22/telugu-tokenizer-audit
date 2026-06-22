from __future__ import annotations

import warnings

import pytest

from telugu_audit.analysis.fertility_accuracy_corr import correlate_fertility_accuracy

# Four models with a clear negative relationship: higher fertility → lower accuracy
_FERTILITY = {"model_a": 2.0, "model_b": 3.0, "model_c": 4.0, "model_d": 5.0}
_ACCURACY  = {"model_a": 0.80, "model_b": 0.70, "model_c": 0.60, "model_d": 0.50}


# ---------------------------------------------------------------------------
# basic correctness
# ---------------------------------------------------------------------------

def test_negative_correlation_detected():
    result = correlate_fertility_accuracy(_FERTILITY, _ACCURACY, n_bootstrap=200)
    assert result["pearson_r"] < 0
    assert result["spearman_r"] < 0


def test_result_keys():
    result = correlate_fertility_accuracy(_FERTILITY, _ACCURACY, n_bootstrap=200)
    assert set(result.keys()) >= {
        "models", "n_models",
        "pearson_r", "pearson_p", "pearson_ci_95",
        "spearman_r", "spearman_p", "spearman_ci_95",
    }


def test_models_list_is_intersection():
    fertility = {"a": 2.0, "b": 3.0, "extra": 4.0}
    accuracy  = {"a": 0.8, "b": 0.6}
    result = correlate_fertility_accuracy(fertility, accuracy, n_bootstrap=100)
    assert result["n_models"] == 2
    assert set(result["models"]) == {"a", "b"}


def test_n_models_matches_shared_count():
    result = correlate_fertility_accuracy(_FERTILITY, _ACCURACY, n_bootstrap=100)
    assert result["n_models"] == len(_FERTILITY)


# ---------------------------------------------------------------------------
# bootstrap CIs
# ---------------------------------------------------------------------------

def test_ci_bounds_ordered():
    result = correlate_fertility_accuracy(_FERTILITY, _ACCURACY, n_bootstrap=500)
    p_lo, p_hi = result["pearson_ci_95"]
    s_lo, s_hi = result["spearman_ci_95"]
    assert p_lo <= p_hi
    assert s_lo <= s_hi


def test_ci_contains_point_estimate():
    result = correlate_fertility_accuracy(_FERTILITY, _ACCURACY, n_bootstrap=1000)
    p_lo, p_hi = result["pearson_ci_95"]
    assert p_lo <= result["pearson_r"] <= p_hi


def test_seed_reproducibility():
    r1 = correlate_fertility_accuracy(_FERTILITY, _ACCURACY, n_bootstrap=300, seed=99)
    r2 = correlate_fertility_accuracy(_FERTILITY, _ACCURACY, n_bootstrap=300, seed=99)
    assert r1["pearson_ci_95"] == r2["pearson_ci_95"]


# ---------------------------------------------------------------------------
# edge cases and warnings
# ---------------------------------------------------------------------------

def test_too_few_models_raises():
    with pytest.raises(ValueError, match="at least 2"):
        correlate_fertility_accuracy({"a": 2.0}, {"a": 0.8}, n_bootstrap=100)


def test_small_n_emits_warning():
    fertility = {"a": 2.0, "b": 3.0, "c": 4.5}  # n=3, below threshold of 5
    accuracy  = {"a": 0.8, "b": 0.7, "c": 0.55}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = correlate_fertility_accuracy(fertility, accuracy, n_bootstrap=200)
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert "warning" in result


def test_small_n_warning_absent_for_large_n():
    # n=4 is still < 5, so warning fires; test that n ≥ 5 suppresses it
    fertility = {f"m{i}": float(i) for i in range(1, 6)}  # 5 models
    accuracy  = {f"m{i}": 1.0 / i for i in range(1, 6)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = correlate_fertility_accuracy(fertility, accuracy, n_bootstrap=200)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0
    assert "warning" not in result
