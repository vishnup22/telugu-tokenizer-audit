import pytest

from telugu_audit.metrics.fertility import byte_count, compute_fertility, word_count


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


def test_compute_fertility_skips_failed_lines():
    call_count = 0

    def fragile_count(text: str) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated tokenizer failure")
        return len(text)

    # three lines; the second call raises — only lines 1 and 3 should count
    lines = ["ab", "cde", "fg"]
    result = compute_fertility(fragile_count, lines)

    assert result["n_lines_attempted"] == 3
    assert result["n_lines_tokenized"] == 2
    # "ab" -> 2 tokens, "cde" skipped, "fg" -> 2 tokens
    assert result["total_tokens"] == 4
    # each remaining line is one whitespace-split word
    assert result["total_words"] == 2
    assert result["fertility_tokens_per_word"] == 2.0
