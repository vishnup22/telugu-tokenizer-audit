from __future__ import annotations


def cost_per_content(n_tokens: int, price_per_1k: float) -> float:
    return n_tokens / 1000.0 * price_per_1k


def effective_context_words(context_limit_tokens: int, fertility: float) -> float:
    if not fertility:
        return 0.0
    return context_limit_tokens / fertility


def cost_and_context_for_model(
    pricing: dict,
    model_name: str,
    n_tokens: int,
    fertility: float,
) -> dict:
    model_cfg = pricing["models"][model_name]
    price = model_cfg["price_per_1k_input_tokens"]
    context_limit = model_cfg["context_limit_tokens"]
    return {
        "model": model_name,
        "cost_usd": cost_per_content(n_tokens, price),
        "effective_context_words": effective_context_words(context_limit, fertility),
        "context_limit_tokens": context_limit,
        "price_per_1k_input_tokens": price,
    }
