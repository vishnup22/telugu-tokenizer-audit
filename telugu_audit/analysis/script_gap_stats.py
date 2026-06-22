from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def per_sentence_gap(
    native_lines: list[str],
    romanized_lines: list[str],
    count_fn,
) -> pd.DataFrame:
    """Per-sentence fertility and gap for matched native/romanized pairs.

    Lines must be parallel — sentence i in native_lines is the same content
    as sentence i in romanized_lines, just in a different script.
    """
    if len(native_lines) != len(romanized_lines):
        raise ValueError(
            f"Line counts differ: {len(native_lines)} native vs "
            f"{len(romanized_lines)} romanized — corpus must be matched pairs"
        )
    rows = []
    for i, (nat, rom) in enumerate(zip(native_lines, romanized_lines)):
        nat_words = len(nat.split())
        rom_words = len(rom.split())
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


def test_gap_significance(
    gap_df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Wilcoxon signed-rank test and bootstrap 95% CI on the median fertility gap.

    Non-parametric because per-sentence fertility is right-skewed.
    H1: native-script fertility > romanized fertility (one-sided).
    Effect size is the common-language statistic: fraction of pairs where
    native costs more tokens per word than romanized.
    """
    gaps = gap_df["fertility_gap"].dropna()
    if len(gaps) < 10:
        raise ValueError(
            f"Too few sentence pairs for a reliable test "
            f"(n={len(gaps)}, need ≥ 10)"
        )

    stat, p_value = stats.wilcoxon(gaps, alternative="greater")

    effect_cl = float((gaps > 0).mean())

    rng = np.random.default_rng(seed)
    boot_medians = np.array(
        [
            np.median(rng.choice(gaps.values, size=len(gaps), replace=True))
            for _ in range(n_bootstrap)
        ]
    )

    return {
        "n_pairs": int(len(gaps)),
        "median_gap": float(gaps.median()),
        "mean_gap": float(gaps.mean()),
        "pct_pairs_native_costlier": round(effect_cl * 100, 1),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
        "bootstrap_ci_95_low": float(np.percentile(boot_medians, 2.5)),
        "bootstrap_ci_95_high": float(np.percentile(boot_medians, 97.5)),
    }
