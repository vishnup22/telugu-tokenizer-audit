#!/usr/bin/env python3
"""Zero-shot Claude Haiku evaluation on MILU Telugu MCQ.

Loads ai4bharat/MILU (Telugu config), runs zero-shot MCQ evaluation using
Claude Haiku, and writes results consumable by 04_correlate_with_benchmarks.py.

Outputs (written to --experiment-dir/results/):
  milu_claude_raw.jsonl          -- one JSON line per question (checkpoint)
  milu_claude_accuracy.json      -- accuracy breakdown by subject/domain + overall
  benchmark_scores_milu.yaml     -- drop-in for 04_correlate_with_benchmarks.py

Prerequisites:
  1. Request MILU access at https://huggingface.co/datasets/ai4bharat/MILU
  2. Set HF_TOKEN in your .env (or environment)
  3. Set ANTHROPIC_API_KEY in your .env (or environment)

Usage:
    python scripts/07_run_milu_eval.py \\
        --experiment-dir experiments/2026-06-23_v3_telugu_gpt2

    # Dry-run on 50 questions to verify setup before full run (~7 k questions):
    python scripts/07_run_milu_eval.py \\
        --experiment-dir experiments/2026-06-23_v3_telugu_gpt2 \\
        --max-questions 50

Cost estimate: full Telugu split (~7 304 questions) ≈ $3-5 at Haiku pricing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

# Load .env before importing anything that reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are answering a multiple-choice exam question. \
Output ONLY a single letter — A, B, C, or D — with no explanation.

{question}

A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Your answer (A, B, C, or D only):"""


def _build_prompt(row: dict) -> tuple[str, str]:
    """Return (prompt_text, correct_letter).

    Handles multiple MILU column-name conventions robustly.
    """
    # --- question text ---
    question = (
        row.get("question")
        or row.get("question_text")
        or row.get("Question")
        or ""
    ).strip()

    # --- options ---
    # Convention A: MILU ai4bharat — option1/option2/option3/option4
    if "option1" in row:
        opt_a = str(row.get("option1", ""))
        opt_b = str(row.get("option2", ""))
        opt_c = str(row.get("option3", ""))
        opt_d = str(row.get("option4", ""))
    # Convention B: option_1 … option_4
    elif "option_1" in row:
        opt_a = str(row.get("option_1", ""))
        opt_b = str(row.get("option_2", ""))
        opt_c = str(row.get("option_3", ""))
        opt_d = str(row.get("option_4", ""))
    # Convention C: options list
    elif "options" in row and isinstance(row["options"], list):
        opts = row["options"]
        while len(opts) < 4:
            opts.append("")
        opt_a, opt_b, opt_c, opt_d = opts[:4]
    # Convention D: option_a … option_d
    else:
        opt_a = str(row.get("option_a", row.get("OptionA", "")))
        opt_b = str(row.get("option_b", row.get("OptionB", "")))
        opt_c = str(row.get("option_c", row.get("OptionC", "")))
        opt_d = str(row.get("option_d", row.get("OptionD", "")))

    prompt = _PROMPT_TEMPLATE.format(
        question=question, opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d
    )

    # --- correct answer → letter ---
    # MILU ai4bharat uses "target": "option1"/"option2"/"option3"/"option4"
    answer_raw = row.get("target", row.get("answer", row.get("correct_option", row.get("Answer", ""))))
    _target_map = {"option1": "A", "option2": "B", "option3": "C", "option4": "D"}
    if isinstance(answer_raw, str) and answer_raw in _target_map:
        correct = _target_map[answer_raw]
    elif isinstance(answer_raw, int):
        correct = "ABCD"[answer_raw - 1] if 1 <= answer_raw <= 4 else "?"
    elif isinstance(answer_raw, str) and answer_raw.strip().upper() in "ABCD":
        correct = answer_raw.strip().upper()
    elif isinstance(answer_raw, str) and answer_raw.strip().isdigit():
        idx = int(answer_raw.strip())
        correct = "ABCD"[idx - 1] if 1 <= idx <= 4 else "?"
    else:
        correct = str(answer_raw).strip()

    return prompt, correct


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_letter(response: str) -> str | None:
    """Extract A/B/C/D from the model response.

    Tries multiple strategies in order:
    1. Response is exactly one letter
    2. First character is the letter
    3. Last character is the letter (CoT that ends with answer)
    4. "answer is X" or "answer: X" pattern
    5. Any standalone A/B/C/D in the uppercased text
    """
    text = response.strip()
    upper = text.upper()

    if not text:
        return None

    # Exact single letter
    if upper in ("A", "B", "C", "D"):
        return upper

    # Starts with the letter (possibly followed by punctuation)
    if upper[0] in "ABCD" and (len(upper) == 1 or not upper[1].isalpha()):
        return upper[0]

    # Ends with the letter (CoT that concludes with answer)
    if upper[-1] in "ABCD" and (len(upper) == 1 or not upper[-2].isalpha()):
        return upper[-1]

    # "answer is X" / "answer: X" / "the answer is X"
    m = re.search(r"answer\s*(?:is|:)?\s*([ABCD])\b", upper)
    if m:
        return m.group(1)

    # "therefore X" / "correct answer is X"
    m = re.search(r"(?:therefore|correct(?:ly)?|option|choice)\s*[:\-]?\s*([ABCD])\b", upper)
    if m:
        return m.group(1)

    # Last resort: any standalone A/B/C/D
    matches = re.findall(r"\b([ABCD])\b", upper)
    if matches:
        return matches[-1]  # take the last one (most likely the conclusion)

    return None


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

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
            print(f"  [retry {attempt+1}/{max_retries}] {exc} — waiting {wait}s")
            time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory (results/ subdir will be used)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap number of questions (omit for full Telugu split)",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model ID to evaluate",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds to sleep between API calls (default 0.1)",
    )
    args = parser.parse_args()

    results_dir = Path(args.experiment_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / "milu_claude_raw.jsonl"
    acc_path = results_dir / "milu_claude_accuracy.json"
    yaml_path = results_dir / "benchmark_scores_milu.yaml"

    # --- load dataset ---
    print("Loading ai4bharat/MILU (Telugu)…")
    try:
        import os
        from datasets import load_dataset
        ds = load_dataset(
            "ai4bharat/MILU",
            "Telugu",
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
        sys.exit(1)

    questions = list(ds)
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"  {len(questions)} questions to evaluate (model: {args.model})")
    print(f"  Checkpoint: {raw_path}")

    # --- resume from checkpoint ---
    done: dict[int, dict] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["idx"]] = rec
        print(f"  Resuming — {len(done)} already done.")

    # --- init anthropic client ---
    import anthropic
    import os
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # --- evaluate ---
    checkpoint_fh = raw_path.open("a", encoding="utf-8")
    try:
        for idx, row in enumerate(questions):
            if idx in done:
                continue

            prompt, correct = _build_prompt(dict(row))
            raw_response = _call_claude(client, prompt, args.model)
            predicted = _parse_letter(raw_response)
            is_correct = predicted == correct if predicted else False

            rec = {
                "idx": idx,
                "subject": row.get("subject", row.get("topic", row.get("category", ""))),
                "domain": row.get("domain", row.get("category", "")),
                "correct": correct,
                "predicted": predicted,
                "raw_response": raw_response,
                "is_correct": is_correct,
            }
            done[idx] = rec
            checkpoint_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            checkpoint_fh.flush()

            if (idx + 1) % 100 == 0:
                so_far = [r for r in done.values() if r["predicted"] is not None]
                acc = sum(r["is_correct"] for r in so_far) / len(so_far) if so_far else 0
                print(f"  [{idx+1}/{len(questions)}] running accuracy: {acc:.3f}")

            time.sleep(args.delay)
    finally:
        checkpoint_fh.close()

    # --- aggregate ---
    records = list(done.values())
    valid = [r for r in records if r["predicted"] is not None]
    unparseable = len(records) - len(valid)

    overall_acc = sum(r["is_correct"] for r in valid) / len(valid) if valid else 0.0

    # per-subject accuracy
    by_subject: dict[str, list[bool]] = {}
    for r in valid:
        subj = r["subject"] or "unknown"
        by_subject.setdefault(subj, []).append(r["is_correct"])
    subject_acc = {k: sum(v) / len(v) for k, v in by_subject.items()}

    # per-domain accuracy
    by_domain: dict[str, list[bool]] = {}
    for r in valid:
        dom = r["domain"] or "unknown"
        by_domain.setdefault(dom, []).append(r["is_correct"])
    domain_acc = {k: sum(v) / len(v) for k, v in by_domain.items()}

    summary = {
        "model": args.model,
        "benchmark": "MILU-Telugu",
        "split": "test",
        "n_questions": len(records),
        "n_valid": len(valid),
        "n_unparseable": unparseable,
        "overall_accuracy": round(overall_acc, 4),
        "accuracy_pct": round(overall_acc * 100, 2),
        "by_subject": {k: round(v, 4) for k, v in sorted(subject_acc.items())},
        "by_domain": {k: round(v, 4) for k, v in sorted(domain_acc.items())},
    }

    with acc_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- write YAML for 04_correlate_with_benchmarks.py ---
    benchmark_yaml = {
        "benchmark": "MILU-Telugu (test, zero-shot)",
        "citation": "Verma et al. 2024 (arXiv:2411.02538)",
        "notes": (
            "GPT-4o and sarvam-2b scores are from Table 3 of the MILU paper (5-shot). "
            "claude score is from our own zero-shot run on the same test split. "
            "Base LMs (hf-gpt2, telugu-gpt2) excluded — no instruction-following capability."
        ),
        "accuracy_by_model": {
            "openai-gpt4o": 72.53,
            "sarvam-2b": 28.57,
            "claude": round(overall_acc * 100, 2),
        },
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(benchmark_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\nDone.")
    print(f"  Overall accuracy : {overall_acc*100:.2f}%  ({len(valid)}/{len(records)} parseable)")
    print(f"  Results written  : {acc_path}")
    print(f"  Benchmark YAML   : {yaml_path}")
    print(f"\nNext step:")
    print(f"  python scripts/04_correlate_with_benchmarks.py \\")
    print(f"    --experiment-dir {args.experiment_dir} \\")
    print(f"    --benchmark-scores {yaml_path}")


if __name__ == "__main__":
    main()
