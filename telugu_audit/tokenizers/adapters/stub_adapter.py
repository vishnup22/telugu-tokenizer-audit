from __future__ import annotations


def get_tokenizers() -> tuple[dict, dict]:
    def count_whitespace(text: str) -> int:
        return len(text.split())

    tokenizers = {"stub-local": count_whitespace}
    versions = {"stub-local": {"library": "builtin", "encoding": "whitespace"}}
    return tokenizers, versions
