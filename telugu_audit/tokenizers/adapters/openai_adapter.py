from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_tokenizers() -> tuple[dict, dict]:
    tokenizers: dict = {}
    versions: dict = {}

    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")

        def count_gpt4o(text: str) -> int:
            return len(enc.encode(text))

        tokenizers["openai-gpt4o"] = count_gpt4o
        versions["openai-gpt4o"] = {
            "library": f"tiktoken=={tiktoken.__version__}",
            "encoding": "o200k_base",
        }
    except Exception as exc:
        logger.warning("Skipping openai-gpt4o: %s", exc)

    return tokenizers, versions


def get_version_info() -> dict:
    _, versions = get_tokenizers()
    return versions
