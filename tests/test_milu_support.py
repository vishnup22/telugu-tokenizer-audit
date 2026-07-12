from __future__ import annotations

import pytest

from telugu_audit.benchmarks.milu import (
    build_milu_output_paths,
    milu_benchmark_name,
    milu_config_name,
    milu_language_slug,
    parse_milu_row,
)


def test_milu_language_mapping_accepts_hindi_and_telugu():
    assert milu_config_name("telugu") == "Telugu"
    assert milu_config_name("hi") == "Hindi"
    assert milu_language_slug("Hindi") == "hindi"
    assert milu_benchmark_name("te") == "MILU-Telugu"


def test_milu_language_mapping_rejects_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported MILU language"):
        milu_config_name("english")


def test_build_output_paths_include_language_slug():
    paths = build_milu_output_paths("results", "claude", "hindi")
    assert paths["raw"].name == "milu_hindi_claude_raw.jsonl"
    assert paths["accuracy"].name == "milu_hindi_claude_accuracy.json"
    assert paths["benchmark_yaml"].name == "benchmark_scores_milu_hindi_claude.yaml"


def test_parse_milu_row_handles_option_columns():
    question, options, correct = parse_milu_row(
        {
            "question": "What?",
            "option1": "one",
            "option2": "two",
            "option3": "three",
            "option4": "four",
            "target": "option3",
        }
    )
    assert question == "What?"
    assert options == {"A": "one", "B": "two", "C": "three", "D": "four"}
    assert correct == "C"


def test_parse_milu_row_handles_list_options():
    question, options, correct = parse_milu_row(
        {
            "question_text": "Pick one",
            "options": ["a", "b", "c", "d"],
            "answer": 2,
        }
    )
    assert question == "Pick one"
    assert options["D"] == "d"
    assert correct == "B"
