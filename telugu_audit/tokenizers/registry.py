from __future__ import annotations

import logging
from collections.abc import Callable

from telugu_audit.tokenizers.adapters import (
    anthropic_adapter,
    hf_adapter,
    openai_adapter,
    stub_adapter,
)

logger = logging.getLogger(__name__)

CountFn = Callable[[str], int]

_ADAPTER_MODULES = [
    stub_adapter,
    openai_adapter,
    anthropic_adapter,
    hf_adapter,
]

_last_version_info: dict[str, dict] = {}


def load_tokenizers(include: set[str] | None = None) -> dict[str, CountFn]:
    global _last_version_info
    merged: dict[str, CountFn] = {}
    versions: dict[str, dict] = {}

    for module in _ADAPTER_MODULES:
        try:
            tokenizers, module_versions = module.get_tokenizers()
        except Exception as exc:
            logger.warning("Adapter %s failed: %s", module.__name__, exc)
            continue
        merged.update(tokenizers)
        versions.update(module_versions)

    if include is not None:
        filtered = {k: v for k, v in merged.items() if k in include}
        _last_version_info = {k: v for k, v in versions.items() if k in include}
        missing = include - set(filtered.keys())
        for name in sorted(missing):
            logger.warning("Requested tokenizer %r could not be loaded", name)
        return filtered

    _last_version_info = versions
    return merged


def get_tokenizer_versions() -> dict[str, dict]:
    return dict(_last_version_info)
