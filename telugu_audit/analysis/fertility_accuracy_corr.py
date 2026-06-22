from __future__ import annotations

from scipy.stats import pearsonr, spearmanr


def correlate_fertility_accuracy(
    fertility_by_model: dict[str, float],
    accuracy_by_model: dict[str, float],
) -> dict:
    shared = sorted(set(fertility_by_model) & set(accuracy_by_model))
    if len(shared) < 2:
        raise ValueError("Need at least 2 models with both fertility and accuracy")

    fertility = [fertility_by_model[m] for m in shared]
    accuracy = [accuracy_by_model[m] for m in shared]

    pearson_r, pearson_p = pearsonr(fertility, accuracy)
    spearman_r, spearman_p = spearmanr(fertility, accuracy)

    return {
        "models": shared,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_models": len(shared),
    }
