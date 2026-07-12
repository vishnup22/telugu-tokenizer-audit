from __future__ import annotations

from pathlib import Path

MILU_LANGUAGE_CONFIGS = {
    "te": "Telugu",
    "telugu": "Telugu",
    "hi": "Hindi",
    "hindi": "Hindi",
}


def normalize_milu_language(language: str) -> str:
    key = language.strip().lower()
    if key not in MILU_LANGUAGE_CONFIGS:
        supported = ", ".join(sorted({"telugu", "hindi"}))
        raise ValueError(
            f"Unsupported MILU language '{language}'. Supported languages: {supported}."
        )
    return key


def milu_config_name(language: str) -> str:
    return MILU_LANGUAGE_CONFIGS[normalize_milu_language(language)]


def milu_language_slug(language: str) -> str:
    key = normalize_milu_language(language)
    return "telugu" if key in {"te", "telugu"} else "hindi"


def milu_benchmark_name(language: str) -> str:
    return f"MILU-{milu_config_name(language)}"


def build_milu_output_paths(
    results_dir: str | Path,
    evaluator_slug: str,
    language: str,
    *,
    legacy_telugu_names: bool = False,
) -> dict[str, Path]:
    results_path = Path(results_dir)
    slug = milu_language_slug(language)

    if legacy_telugu_names and slug == "telugu":
        raw_name = f"milu_{evaluator_slug}_raw.jsonl"
        acc_name = f"milu_{evaluator_slug}_accuracy.json"
        yaml_name = f"benchmark_scores_milu_{evaluator_slug}.yaml"
    else:
        raw_name = f"milu_{slug}_{evaluator_slug}_raw.jsonl"
        acc_name = f"milu_{slug}_{evaluator_slug}_accuracy.json"
        yaml_name = f"benchmark_scores_milu_{slug}_{evaluator_slug}.yaml"

    return {
        "raw": results_path / raw_name,
        "accuracy": results_path / acc_name,
        "benchmark_yaml": results_path / yaml_name,
    }


def parse_milu_row(row: dict) -> tuple[str, dict[str, str], str]:
    question = (
        row.get("question")
        or row.get("question_text")
        or row.get("Question")
        or ""
    ).strip()

    if "option1" in row:
        options = {
            "A": str(row.get("option1", "")),
            "B": str(row.get("option2", "")),
            "C": str(row.get("option3", "")),
            "D": str(row.get("option4", "")),
        }
    elif "option_1" in row:
        options = {
            "A": str(row.get("option_1", "")),
            "B": str(row.get("option_2", "")),
            "C": str(row.get("option_3", "")),
            "D": str(row.get("option_4", "")),
        }
    elif "options" in row and isinstance(row["options"], list):
        padded = (list(row["options"]) + [""] * 4)[:4]
        options = dict(zip(["A", "B", "C", "D"], map(str, padded)))
    else:
        options = {
            "A": str(row.get("option_a", row.get("OptionA", ""))),
            "B": str(row.get("option_b", row.get("OptionB", ""))),
            "C": str(row.get("option_c", row.get("OptionC", ""))),
            "D": str(row.get("option_d", row.get("OptionD", ""))),
        }

    target_map = {"option1": "A", "option2": "B", "option3": "C", "option4": "D"}
    raw = row.get(
        "target",
        row.get("answer", row.get("correct_option", row.get("Answer", ""))),
    )
    if isinstance(raw, str) and raw in target_map:
        correct = target_map[raw]
    elif isinstance(raw, int) and 1 <= raw <= 4:
        correct = "ABCD"[raw - 1]
    elif isinstance(raw, str) and raw.strip().upper() in "ABCD":
        correct = raw.strip().upper()
    elif isinstance(raw, str) and raw.strip().isdigit():
        idx = int(raw.strip())
        correct = "ABCD"[idx - 1] if 1 <= idx <= 4 else "?"
    else:
        correct = str(raw).strip()

    return question, options, correct


def row_subject(row: dict) -> str:
    return row.get("subject", row.get("topic", row.get("category", ""))) or "unknown"


def row_domain(row: dict) -> str:
    return row.get("domain", row.get("category", "")) or "unknown"
