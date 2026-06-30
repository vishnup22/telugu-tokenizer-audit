import pytest

from telugu_audit.metrics.fertility import (
    byte_count,
    compute_fertility,
    get_word_count_fn,
    word_count,
)


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("  ") == 0


def test_byte_count():
    assert byte_count("abc") == 3


def test_compute_fertility():
    def fake_count(text: str) -> int:
        return len(text)

    lines = ["ab", "cd"]
    result = compute_fertility(fake_count, lines)

    assert result["n_lines_attempted"] == 2
    assert result["n_lines_tokenized"] == 2
    assert result["total_tokens"] == 4
    assert result["total_words"] == 2
    assert result["fertility_tokens_per_word"] == 2.0
    assert result["compression_bytes_per_token"] == 1.0
    assert result["word_count_method"] == "whitespace"


def test_compute_fertility_skips_failed_lines():
    call_count = 0

    def fragile_count(text: str) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated tokenizer failure")
        return len(text)

    lines = ["ab", "cde", "fg"]
    result = compute_fertility(fragile_count, lines)

    assert result["n_lines_attempted"] == 3
    assert result["n_lines_tokenized"] == 2
    assert result["total_tokens"] == 4
    assert result["total_words"] == 2
    assert result["fertility_tokens_per_word"] == 2.0


def test_compute_fertility_uses_custom_word_counter():
    def fake_count(text: str) -> int:
        return len(text.split())

    def doubled_word_count(_: str) -> int:
        return 2

    result = compute_fertility(
        fake_count,
        ["a b", "c d"],
        word_count_fn=doubled_word_count,
        word_count_method="custom",
    )

    assert result["total_tokens"] == 4
    assert result["total_words"] == 4
    assert result["fertility_tokens_per_word"] == 1.0
    assert result["word_count_method"] == "custom"


def test_get_word_count_fn_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown word-count method"):
        get_word_count_fn("bogus")
