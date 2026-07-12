                      
"""Zero-shot Claude evaluation on MILU MCQ for a supported language."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.benchmarks.milu import (
    build_milu_output_paths,
    milu_benchmark_name,
    milu_config_name,
    milu_language_slug,
    parse_milu_row,
    row_domain,
    row_subject,
)

_PROMPT_TEMPLATE = """\
You are answering a multiple-choice exam question. \
Output ONLY a single letter - A, B, C, or D - with no explanation.

{question}

A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Your answer (A, B, C, or D only):"""


def _build_prompt(row: dict) -> tuple[str, str]:
    question, options, correct = parse_milu_row(row)
    prompt = _PROMPT_TEMPLATE.format(
        question=question,
        opt_a=options["A"],
        opt_b=options["B"],
        opt_c=options["C"],
        opt_d=options["D"],
    )
    return prompt, correct


def _parse_letter(response: str) -> str | None:
    text = response.strip()
    upper = text.upper()

    if not text:
        return None
    if upper in ("A", "B", "C", "D"):
        return upper
    if upper[0] in "ABCD" and (len(upper) == 1 or not upper[1].isalpha()):
        return upper[0]
    if upper[-1] in "ABCD" and (len(upper) == 1 or not upper[-2].isalpha()):
        return upper[-1]

    match = re.search(r"answer\s*(?:is|:)?\s*([ABCD])\b", upper)
    if match:
        return match.group(1)

    match = re.search(r"(?:therefore|correct(?:ly)?|option|choice)\s*[:\-]?\s*([ABCD])\b", upper)
    if match:
        return match.group(1)

    matches = re.findall(r"\b([ABCD])\b", upper)
    return matches[-1] if matches else None


def _call_claude(client, prompt: str, model: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=50,
                system="You are a multiple-choice exam assistant. Always respond with exactly one letter: A, B, C, or D. Never explain.",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "The answer is"},
                ],
            )
            return msg.content[0].text
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {exc} - waiting {wait}s")
            time.sleep(wait)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--language", default="telugu", help="Supported: telugu, hindi")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    results_dir = Path(args.experiment_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    language_label = milu_config_name(args.language)
    language_slug = milu_language_slug(args.language)
    paths = build_milu_output_paths(results_dir, "claude", args.language)
    raw_path = paths["raw"]
    acc_path = paths["accuracy"]
    yaml_path = paths["benchmark_yaml"]

    print(f"Loading ai4bharat/MILU ({language_label})...")
    try:
        import os
        from datasets import load_dataset

        ds = load_dataset(
            "ai4bharat/MILU",
            language_label,
            split="test",
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        if "gated" in str(exc).lower() or "access" in str(exc).lower() or "401" in str(exc):
            print(
                "\nERROR: MILU is a gated dataset.\n"
                "  1. Visit https://huggingface.co/datasets/ai4bharat/MILU and request access.\n"
                "  2. Once approved, make sure HF_TOKEN is set in your .env file.\n"
            )
        else:
            print(f"\nERROR loading dataset: {exc}\n")
        raise SystemExit(1)

    questions = list(ds)
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"  {len(questions)} questions to evaluate (language: {language_label}, model: {args.model})")
    print(f"  Checkpoint: {raw_path}")

    done: dict[int, dict] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                done[record["idx"]] = record
        print(f"  Resuming - {len(done)} already done.")

    import anthropic
    import os

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    checkpoint_handle = raw_path.open("a", encoding="utf-8")
    try:
        for idx, row in enumerate(questions):
            if idx in done:
                continue

            prompt, correct = _build_prompt(dict(row))
            raw_response = _call_claude(client, prompt, args.model)
            predicted = _parse_letter(raw_response)
            is_correct = predicted == correct if predicted else False

            record = {
                "idx": idx,
                "language": language_slug,
                "subject": row_subject(row),
                "domain": row_domain(row),
                "correct": correct,
                "predicted": predicted,
                "raw_response": raw_response,
                "is_correct": is_correct,
            }
            done[idx] = record
            checkpoint_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint_handle.flush()

            if (idx + 1) % 100 == 0:
                so_far = [item for item in done.values() if item["predicted"] is not None]
                accuracy = sum(item["is_correct"] for item in so_far) / len(so_far) if so_far else 0.0
                print(f"  [{idx + 1}/{len(questions)}] running accuracy: {accuracy:.3f}")

            time.sleep(args.delay)
    finally:
        checkpoint_handle.close()

    records = list(done.values())
    valid = [record for record in records if record["predicted"] is not None]
    unparseable = len(records) - len(valid)
    overall_accuracy = sum(record["is_correct"] for record in valid) / len(valid) if valid else 0.0

    by_subject: dict[str, list[bool]] = {}
    by_domain: dict[str, list[bool]] = {}
    for record in valid:
        by_subject.setdefault(record["subject"], []).append(record["is_correct"])
        by_domain.setdefault(record["domain"], []).append(record["is_correct"])

    summary = {
        "model": args.model,
        "benchmark": milu_benchmark_name(args.language),
        "language": language_slug,
        "split": "test",
        "n_questions": len(records),
        "n_valid": len(valid),
        "n_unparseable": unparseable,
        "overall_accuracy": round(overall_accuracy, 4),
        "accuracy_pct": round(overall_accuracy * 100, 2),
        "by_subject": {key: round(sum(values) / len(values), 4) for key, values in sorted(by_subject.items())},
        "by_domain": {key: round(sum(values) / len(values), 4) for key, values in sorted(by_domain.items())},
    }

    with acc_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    benchmark_yaml = {
        "benchmark": f"{milu_benchmark_name(args.language)} (test, zero-shot)",
        "citation": "Verma et al. 2024 (arXiv:2411.02538)",
        "notes": (
            "Claude score is from our own zero-shot run on the same test split. "
            "Published comparison points should be added per language before correlation."
        ),
        "accuracy_by_model": {
            "claude": round(overall_accuracy * 100, 2),
        },
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.dump(benchmark_yaml, handle, allow_unicode=True, sort_keys=False)

    print("\nDone.")
    print(f"  Overall accuracy : {overall_accuracy * 100:.2f}%  ({len(valid)}/{len(records)} parseable)")
    print(f"  Results written  : {acc_path}")
    print(f"  Benchmark YAML   : {yaml_path}")


if __name__ == "__main__":
    main()
