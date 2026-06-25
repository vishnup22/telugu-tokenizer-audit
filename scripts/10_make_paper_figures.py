#!/usr/bin/env python3
"""Generate all publication-ready figures for the MRL @ EMNLP 2026 paper.

Produces five PDFs in <experiment-dir>/figures/:
  fig1_fertility_bar.pdf
  fig2_gap_violin.pdf
  fig3_heatmap_morph.pdf
  fig4_token_length_dist.pdf
  fig5_fertility_vs_accuracy.pdf

Usage:
    python scripts/10_make_paper_figures.py \
        --experiment-dir experiments/2026-06-23_v3_telugu_gpt2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size":          7,
    "axes.titlesize":     7,
    "axes.labelsize":     7,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "legend.fontsize":    6.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.4,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ACL two-column dimensions in inches
COL_W  = 3.33   # single column width
TEXT_W = 6.97   # full text width (both columns)

# Colorblind-safe (Wong 2011)
COLORS = {
    "native_formal":      "#0072B2",
    "native_informal":    "#E69F00",
    "romanized_informal": "#009E73",
}

TOKENIZER_ORDER = [
    "telugu-gpt2",
    "sarvam-2b",
    "sarvam-105b",
    "openai-gpt4o",
    "claude",
    "hf-gpt2",
]

TOKENIZER_LABELS = {
    "telugu-gpt2":   "dravidian-\ngpt2-telugu",
    "sarvam-2b":     "Sarvam-2B",
    "sarvam-105b":   "Sarvam-105B",
    "openai-gpt4o":  "GPT-4o",
    "claude":        "Claude\nHaiku 4.5",
    "hf-gpt2":       "GPT-2\n(legacy)",
}

TOKENIZER_LABELS_FLAT = {
    "telugu-gpt2":   "dravidian-gpt2-telugu",
    "sarvam-2b":     "Sarvam-2B",
    "sarvam-105b":   "Sarvam-105B",
    "openai-gpt4o":  "GPT-4o",
    "claude":        "Claude Haiku 4.5",
    "hf-gpt2":       "GPT-2 (legacy)",
}

MORPH_LABELS = {
    "base_noun":                "Base noun",
    "borrowed_base":            "Borrowed base",
    "borrowed_plus_case_suffix":"Borrowed+case",
    "case_suffix":              "Case suffix",
    "case_suffix_with_sandhi":  "Case+sandhi",
    "compound_with_sandhi":     "Compound+sandhi",
    "honorific_suffix":         "Honorific suffix",
    "plural_suffix":            "Plural suffix",
    "verb_agglutination_chain": "Verb agglutination",
}

REGISTER_LABELS = {
    "native_formal":      "Native formal",
    "native_informal":    "Native informal",
    "romanized_informal": "Romanized informal",
}


# ---------------------------------------------------------------------------
# Figure 1 — Grouped bar: fertility × register × tokenizer
# ---------------------------------------------------------------------------

def fig1_fertility_bar(df: pd.DataFrame, out: Path) -> None:
    registers = ["native_formal", "native_informal", "romanized_informal"]
    n_tok = len(TOKENIZER_ORDER)
    bar_w = 0.24
    gap   = 0.08
    x = np.arange(n_tok) * (len(registers) * bar_w + gap)

    fig, ax = plt.subplots(figsize=(TEXT_W, 2.4))

    for i, reg in enumerate(registers):
        vals = []
        for tok in TOKENIZER_ORDER:
            row = df[(df["tokenizer"] == tok) & (df["register"] == reg)]
            vals.append(float(row["fertility_tokens_per_word"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + i * bar_w, vals, width=bar_w,
               color=COLORS[reg], label=REGISTER_LABELS[reg],
               edgecolor="white", linewidth=0.4)
        # annotate only GPT-2 bars (tallest)
        for xi, val in zip(x + i * bar_w, vals):
            if val > 15:
                ax.text(xi, val + 0.3, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=8.5, rotation=90)

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels([TOKENIZER_LABELS[t] for t in TOKENIZER_ORDER],
                       ha="center", linespacing=1.2)
    ax.set_ylabel("Fertility (tokens / word)")
    ax.set_ylim(0, 27)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 2 — Violin: per-sentence δ distribution
# ---------------------------------------------------------------------------

def fig2_gap_violin(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.6))

    data, labels = [], []
    native_costlier  = "#E69F00"
    roman_costlier   = "#009E73"
    gap_colors = [roman_costlier, roman_costlier, roman_costlier,
                  roman_costlier, native_costlier, native_costlier]

    for tok in TOKENIZER_ORDER:
        gaps = df[df["tokenizer"] == tok]["fertility_gap"].values
        data.append(np.clip(gaps, -6, 30))
        labels.append(TOKENIZER_LABELS[tok])

    parts = ax.violinplot(data, positions=range(len(TOKENIZER_ORDER)),
                          showmedians=True, showextrema=False, widths=0.7)

    for pc, col in zip(parts["bodies"], gap_colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
        pc.set_edgecolor("0.35")
        pc.set_linewidth(0.8)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.axhline(0, color="0.4", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xticks(range(len(TOKENIZER_ORDER)))
    ax.set_xticklabels(labels, ha="center", linespacing=1.2)
    ax.set_ylabel("Per-sentence gap δ (tokens / word)\nnative − romanized")
    ax.set_ylim(-7, 30)

    patch_pos = mpatches.Patch(color=native_costlier, alpha=0.75,
                               label="Native costlier (δ > 0)")
    patch_neg = mpatches.Patch(color=roman_costlier,  alpha=0.75,
                               label="Romanized costlier (δ < 0)")
    ax.legend(handles=[patch_pos, patch_neg], loc="upper left", frameon=False)
    ax.text(0.99, 0.02, "GPT-2 clipped to [−6, 30] for scale",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.5", style="italic")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 — Heatmap: tokens/word × morph type × tokenizer
# ---------------------------------------------------------------------------

def fig3_heatmap_morph(df: pd.DataFrame, out: Path) -> None:
    from matplotlib.colors import LogNorm

    morph_order = [
        "base_noun", "borrowed_base", "borrowed_plus_case_suffix",
        "case_suffix", "case_suffix_with_sandhi", "compound_with_sandhi",
        "plural_suffix", "verb_agglutination_chain", "honorific_suffix",
    ]
    tok_order = ["telugu-gpt2", "sarvam-2b", "sarvam-105b",
                 "openai-gpt4o", "claude", "hf-gpt2"]

    matrix = np.zeros((len(morph_order), len(tok_order)))
    for i, morph in enumerate(morph_order):
        for j, tok in enumerate(tok_order):
            row = df[(df["tokenizer"] == tok) & (df["morph_type"] == morph)]
            if len(row):
                matrix[i, j] = float(row["mean_tokens"].iloc[0])

    fig, ax = plt.subplots(figsize=(TEXT_W, 3.2))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   norm=LogNorm(vmin=1.0, vmax=matrix.max()))

    for i in range(len(morph_order)):
        for j in range(len(tok_order)):
            val = matrix[i, j]
            color = "white" if val > 14 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9.5, color=color, fontweight="bold")

    ax.set_xticks(range(len(tok_order)))
    ax.set_xticklabels(
        [TOKENIZER_LABELS_FLAT[t] for t in tok_order],
        rotation=25, ha="right", fontsize=10,
    )
    ax.set_yticks(range(len(morph_order)))
    ax.set_yticklabels([MORPH_LABELS[m] for m in morph_order], fontsize=10)
    ax.tick_params(axis="x", pad=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean tokens / word", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 4 — Stacked bar: token-length distribution (native informal)
# ---------------------------------------------------------------------------

def fig4_token_length_dist(df: pd.DataFrame, out: Path) -> None:
    register = "native_informal"
    df = df[df["register"] == register].copy()
    df["tokens_per_word"] = df["tokens_per_word"].astype(str)

    bins       = ["1", "2", "3", "4", "5+"]
    bin_colors = ["#0072B2", "#56B4E9", "#009E73", "#F0E442", "#E69F00"]
    bin_labels = ["1 token", "2 tokens", "3 tokens", "4 tokens", "5+ tokens"]

    pct = np.zeros((len(TOKENIZER_ORDER), len(bins)))
    for i, tok in enumerate(TOKENIZER_ORDER):
        sub = df[df["tokenizer"] == tok]
        for j, b in enumerate(bins):
            row = sub[sub["tokens_per_word"] == b]
            if len(row):
                pct[i, j] = float(row["pct_words"].iloc[0])

    fig, ax = plt.subplots(figsize=(TEXT_W, 2.4))
    bottoms = np.zeros(len(TOKENIZER_ORDER))
    bar_w = 0.55

    for j, (b, col, lbl) in enumerate(zip(bins, bin_colors, bin_labels)):
        ax.bar(range(len(TOKENIZER_ORDER)), pct[:, j],
               bottom=bottoms, width=bar_w,
               color=col, label=lbl,
               edgecolor="white", linewidth=0.4)
        bottoms += pct[:, j]

    ax.set_xticks(range(len(TOKENIZER_ORDER)))
    ax.set_xticklabels([TOKENIZER_LABELS[t] for t in TOKENIZER_ORDER],
                       ha="center", linespacing=1.2)
    ax.set_ylabel("% of words")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13),
              ncol=5, frameon=False, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Figure 5 — Scatter: fertility vs MILU accuracy
# ---------------------------------------------------------------------------

def fig5_fertility_vs_accuracy(df: pd.DataFrame,
                                milu_claude: dict,
                                milu_dravidian: dict,
                                out: Path) -> None:
    def _fert(tok):
        row = df[(df["tokenizer"] == tok) & (df["register"] == "native_informal")]
        return float(row["fertility_tokens_per_word"].iloc[0]) if len(row) else None

    points = [
        dict(label="GPT-4o\n(paper, 5-shot)",
             fertility=_fert("openai-gpt4o"), accuracy=72.53,
             color="#0072B2", marker="s",
             xytext=(4.2, 68)),
        dict(label="Claude Haiku 4.5\n(ours, 0-shot)",
             fertility=_fert("claude"),
             accuracy=milu_claude.get("accuracy_pct", 60.78),
             color="#E69F00", marker="o",
             xytext=(6.0, 52)),
        dict(label="dravidian-gpt2-telugu\n(ours, log-likelihood)",
             fertility=_fert("telugu-gpt2"),
             accuracy=milu_dravidian.get("accuracy_pct", 26.97),
             color="#009E73", marker="^",
             xytext=(2.8, 14)),
    ]

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    for pt in points:
        ax.scatter(pt["fertility"], pt["accuracy"],
                   color=pt["color"], marker=pt["marker"],
                   s=40, zorder=3, edgecolors="0.3", linewidths=0.5)
        ax.annotate(pt["label"],
                    xy=(pt["fertility"], pt["accuracy"]),
                    xytext=pt["xytext"],
                    fontsize=5.5,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5))

    ax.axhline(25.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.text(8.5, 26.5, "Chance (25%)", fontsize=5.5, color="0.5", ha="right")

    ax.set_xlabel("Native informal fertility (tokens / word)")
    ax.set_ylabel("MILU Telugu accuracy (%)")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 88)
    ax.text(0.98, 0.04, "Lower fertility $\\neq$ higher accuracy",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.5, style="italic", color="0.45")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()

    exp     = Path(args.experiment_dir)
    results = exp / "results"
    figures = exp / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df_fertility = pd.read_csv(results / "fertility_by_register.csv")
    df_gaps      = pd.read_csv(results / "per_sentence_gaps.csv")
    df_morph     = pd.read_csv(results / "minimal_pair_summary.csv")
    df_tld       = pd.read_csv(results / "token_length_dist.csv")

    with open(results / "milu_claude_accuracy.json", encoding="utf-8") as f:
        milu_claude = json.load(f)
    with open(results / "milu_dravidian_accuracy.json", encoding="utf-8") as f:
        milu_dravidian = json.load(f)

    print("Generating figures …")
    fig1_fertility_bar(df_fertility,   figures / "fig1_fertility_bar.pdf")
    fig2_gap_violin(df_gaps,           figures / "fig2_gap_violin.pdf")
    fig3_heatmap_morph(df_morph,       figures / "fig3_heatmap_morph.pdf")
    fig4_token_length_dist(df_tld,     figures / "fig4_token_length_dist.pdf")
    fig5_fertility_vs_accuracy(
        df_fertility, milu_claude, milu_dravidian,
        figures / "fig5_fertility_vs_accuracy.pdf",
    )
    print(f"\nDone -> {figures}/")


if __name__ == "__main__":
    main()
