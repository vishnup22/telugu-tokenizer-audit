from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr


def correlate_fertility_accuracy(
    fertility_by_model: dict[str, float],
    accuracy_by_model: dict[str, float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Pearson and Spearman correlation with bootstrap 95% CIs.

    With only 4-5 tokenizers, p-values alone are unreliable — report CIs
    alongside them and let the reader judge practical significance.
    """
    shared = sorted(set(fertility_by_model) & set(accuracy_by_model))
    if len(shared) < 2:
        raise ValueError("Need at least 2 models with both fertility and accuracy")

    fertility = np.array([fertility_by_model[m] for m in shared])
    accuracy = np.array([accuracy_by_model[m] for m in shared])

    pearson_r, pearson_p = pearsonr(fertility, accuracy)
    spearman_r, spearman_p = spearmanr(fertility, accuracy)

    rng = np.random.default_rng(seed)
    n = len(shared)
    boot_pearson: list[float] = []
    boot_spearman: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f_b, a_b = fertility[idx], accuracy[idx]
        if np.std(f_b) == 0 or np.std(a_b) == 0:
            continue
        boot_pearson.append(float(pearsonr(f_b, a_b)[0]))
        boot_spearman.append(float(spearmanr(f_b, a_b)[0]))

    def _ci(samples: list[float]) -> list[float]:
        if len(samples) < 100:
            return [float("nan"), float("nan")]
        return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]

    result: dict = {
        "models": shared,
        "n_models": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "pearson_ci_95": _ci(boot_pearson),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "spearman_ci_95": _ci(boot_spearman),
    }

    if n < 5:
        msg = (
            f"Only {n} models — bootstrap CIs are wide and p-values should "
            "be interpreted cautiously; report CIs in the paper, not just p-values"
        )
        result["warning"] = msg
        warnings.warn(msg, UserWarning, stacklevel=2)

    return result
