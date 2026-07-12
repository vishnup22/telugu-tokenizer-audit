from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_fertility_by_register(results_dir: str | Path, figures_dir: str | Path) -> None:
    path = Path(results_dir) / "fertility_by_register.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)

    for register in df["register"].unique():
        subset = df[df["register"] == register]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(subset["tokenizer"], subset["fertility_tokens_per_word"])
        ax.set_title(f"Fertility by tokenizer ({register})")
        ax.set_ylabel("tokens per word")
        ax.tick_params(axis="x", rotation=45)
        _save_figure(fig, out, f"fertility_by_register_{register}")


def plot_script_fairness_gap(results_dir: str | Path, figures_dir: str | Path) -> None:
    path = Path(results_dir) / "script_fairness_gap.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    if df.empty:
        return

    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "language" in df.columns and df["language"].nunique() > 1:
        for language, subset in df.groupby("language"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(subset["tokenizer"], subset["script_fertility_ratio_native_over_romanized"])
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(f"Script fairness gap ({language}: native / romanized fertility)")
            ax.set_ylabel("ratio")
            ax.tick_params(axis="x", rotation=45)
            _save_figure(fig, out, f"script_fairness_gap_{language}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["tokenizer"], df["script_fertility_ratio_native_over_romanized"])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Script fairness gap (native / romanized fertility)")
    ax.set_ylabel("ratio")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, out, "script_fairness_gap")


def plot_fertility_accuracy_scatter(
    fertility_by_model: dict[str, float],
    accuracy_by_model: dict[str, float],
    figures_dir: str | Path,
) -> None:
    shared = sorted(set(fertility_by_model) & set(accuracy_by_model))
    if len(shared) < 2:
        return

    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = [fertility_by_model[m] for m in shared]
    y = [accuracy_by_model[m] for m in shared]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y)
    for model, xi, yi in zip(shared, x, y):
        ax.annotate(model, (xi, yi), fontsize=8)
    ax.set_xlabel("fertility (tokens per word)")
    ax.set_ylabel("benchmark accuracy")
    ax.set_title("Fertility vs. accuracy")
    _save_figure(fig, out, "fertility_vs_accuracy")


def export_all_figures(experiment_dir: str | Path) -> None:
    exp = Path(experiment_dir)
    results_dir = exp / "results"
    figures_dir = exp / "figures"
    plot_fertility_by_register(results_dir, figures_dir)
    plot_script_fairness_gap(results_dir, figures_dir)
