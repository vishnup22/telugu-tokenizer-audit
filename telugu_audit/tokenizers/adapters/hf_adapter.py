from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

HF_MODELS = {
    "hf-gpt2": "gpt2",
}


def get_tokenizers() -> tuple[dict, dict]:
    tokenizers: dict = {}
    versions: dict = {}

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.warning("Skipping HF tokenizers: %s", exc)
        return tokenizers, versions

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    for name, model_id in HF_MODELS.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

            def make_counter(tok):
                def count(text: str) -> int:
                    return len(tok.encode(text, add_special_tokens=False))

                return count

            tokenizers[name] = make_counter(tokenizer)
            versions[name] = {
                "library": f"transformers=={transformers.__version__}",
                "model": model_id,
                "revision": getattr(tokenizer, "_commit_hash", None) or "unknown",
            }
        except Exception as exc:
            logger.warning("Skipping %s: %s", name, exc)

    return tokenizers, versions
