from __future__ import annotations

from pathlib import Path

import yaml


def load_pricing(pricing_path: str | Path) -> dict:
    with Path(pricing_path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def cost_per_content(n_tokens: int, price_per_1k_tokens: float) -> float:
    return (n_tokens / 1000.0) * price_per_1k_tokens


def effective_context_words(context_limit_tokens: int, fertility: float) -> float:
    if fertility == 0:
        return 0.0
    return context_limit_tokens / fertility


def cost_and_context_for_model(
    pricing: dict,
    model_name: str,
    n_tokens: int,
    fertility: float,
) -> dict:
    models = pricing.get("models", {})
    if model_name not in models:
        raise KeyError(f"No pricing entry for model {model_name!r}")

    entry = models[model_name]
    price = entry["price_per_1k_input_tokens"]
    context_limit = entry["context_limit_tokens"]

    return {
        "model": model_name,
        "cost_usd": cost_per_content(n_tokens, price),
        "effective_context_words": effective_context_words(context_limit, fertility),
        "context_limit_tokens": context_limit,
        "price_per_1k_input_tokens": price,
    }
