from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DISK_CACHE_PATH = Path(__file__).resolve().parents[4] / ".claude_token_cache.json"


def _load_disk_cache() -> dict[str, int]:
    if _DISK_CACHE_PATH.exists():
        try:
            return json.loads(_DISK_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_disk_cache(cache: dict[str, int]) -> None:
    try:
        _DISK_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as exc:
        logger.debug("Could not save disk cache: %s", exc)


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
        cache: dict[str, int] = _load_disk_cache()
        logger.info("Claude disk cache: %d entries pre-loaded", len(cache))

                                                                           
                                                                             
                                                                          
                                           
        _probe_resp = client.messages.count_tokens(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "a"}],
        )
        _msg_overhead = _probe_resp.input_tokens - 1

        _save_counter = [0]

        def count_claude(text: str) -> int:
            if text in cache:
                return cache[text]
            resp = client.messages.count_tokens(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": text}],
            )
            count = max(0, resp.input_tokens - _msg_overhead)
            cache[text] = count
            _save_counter[0] += 1
            if _save_counter[0] % 100 == 0:
                _save_disk_cache(cache)
            return count

        import atexit
        atexit.register(_save_disk_cache, cache)

        tokenizers["claude"] = count_claude
        versions["claude"] = {
            "library": f"anthropic=={anthropic.__version__}",
            "model": DEFAULT_MODEL,
            "method": "messages.count_tokens API",
        }
    except Exception as exc:
        logger.warning("Skipping claude: %s", exc)

    return tokenizers, versions
