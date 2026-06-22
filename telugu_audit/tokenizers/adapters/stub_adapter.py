from __future__ import annotations


def get_tokenizers() -> tuple[dict, dict]:
    # deterministic stand-in for CI — not a real tokenizer
    def count_stub(text: str) -> int:
        if not text.strip():
            return 0
        return max(1, len(text) // 4)

    tokenizers = {"stub-local": count_stub}
    versions = {
        "stub-local": {
            "library": "telugu_audit (built-in stub)",
            "method": "len(text)//4",
        }
    }
    return tokenizers, versions
