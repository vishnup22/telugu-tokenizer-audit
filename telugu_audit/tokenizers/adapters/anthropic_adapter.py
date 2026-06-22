from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-3-5-haiku-20241022"


def get_tokenizers() -> tuple[dict, dict]:
    tokenizers: dict = {}
    versions: dict = {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("Skipping claude: ANTHROPIC_API_KEY not set")
        return tokenizers, versions

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        cache: dict[str, int] = {}

        def count_claude(text: str) -> int:
            if text in cache:
                return cache[text]
            resp = client.messages.count_tokens(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": text}],
            )
            count = resp.input_tokens
            cache[text] = count
            return count

        tokenizers["claude"] = count_claude
        versions["claude"] = {
            "library": f"anthropic=={anthropic.__version__}",
            "model": DEFAULT_MODEL,
            "method": "messages.count_tokens API",
        }
    except Exception as exc:
        logger.warning("Skipping claude: %s", exc)

    return tokenizers, versions
