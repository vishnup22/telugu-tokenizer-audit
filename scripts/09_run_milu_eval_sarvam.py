#!/usr/bin/env python3
"""Zero-shot generative evaluation of Sarvam-2B on MILU Telugu.

Sarvam-2B is an instruction-tuned model, so we run it the same way as the
Claude eval (zero-shot generative MCQ) but using the local HF model instead
of an API — no cost, runs fully on Colab/GPU.

The model is loaded in bfloat16 and uses greedy decoding (max_new_tokens=10).
We apply the Sarvam chat template if available, falling back to a plain
instruction prompt.

Outputs (experiment-dir/results/):
  milu_sarvam_raw.jsonl             -- one JSON line per question (checkpoint)
  milu_sarvam_accuracy.json         -- overall + per-subject breakdown
  benchmark_scores_milu_sarvam.yaml -- for 04_correlate_with_benchmarks.py

Usage:
    python scripts/09_run_milu_eval_sarvam.py \\
        --experiment-dir experiments/2026-06-23_v3_telugu_gpt2 \\
        --device cuda

    # Smoke-test on 20 questions first:
    python scripts/09_run_milu_eval_sarvam.py \\
        --experiment-dir experiments/2026-06-23_v3_telugu_gpt2 \\
        --device cuda --max-questions 20

Requires: transformers>=4.40, torch, datasets, huggingface_hub, pyyaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_ID = "sarvamai/sarvam-1-v0.5"
MODEL_LABEL = "sarvam-2b"

_PROMPT_TEMPLATE = """\
You are answering a multiple-choice exam question. \
Output ONLY a single letter — A, B, C, or D — with no explanation.

{question}

A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Your answer (A, B, C, or D only):"""


# ---------------------------------------------------------------------------
# Row parsing  (identical to scripts/07 and 08)
# ---------------------------------------------------------------------------

def _parse_row(row: dict) -> tuple[str, str]:
    """Return (prompt_text, correct_letter)."""
    question = (
        row.get("question") or row.get("question_text") or row.get("Question") or ""
    ).strip()

    if "option1" in row:
        opt_a = str(row.get("option1", ""))
        opt_b = str(row.get("option2", ""))
        opt_c = str(row.get("option3", ""))
        opt_d = str(row.get("option4", ""))
    elif "option_1" in row:
        opt_a = str(row.get("option_1", ""))
        opt_b = str(row.get("option_2", ""))
        opt_c = str(row.get("option_3", ""))
        opt_d = str(row.get("option_4", ""))
    elif "options" in row and isinstance(row["options"], list):
        opts = (list(row["options"]) + [""] * 4)[:4]
        opt_a, opt_b, opt_c, opt_d = opts
    else:
        opt_a = str(row.get("option_a", row.get("OptionA", "")))
        opt_b = str(row.get("option_b", row.get("OptionB", "")))
        opt_c = str(row.get("option_c", row.get("OptionC", "")))
        opt_d = str(row.get("option_d", row.get("OptionD", "")))

    prompt = _PROMPT_TEMPLATE.format(
        question=question, opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d
    )

    _tmap = {"option1": "A", "option2": "B", "option3": "C", "option4": "D"}
    raw = row.get("target", row.get("answer", row.get("correct_option", row.get("Answer", ""))))
    if isinstance(raw, str) and raw in _tmap:
        correct = _tmap[raw]
    elif isinstance(raw, int) and 1 <= raw <= 4:
        correct = "ABCD"[raw - 1]
    elif isinstance(raw, str) and raw.strip().upper() in "ABCD":
        correct = raw.strip().upper()
    elif isinstance(raw, str) and raw.strip().isdigit():
        idx = int(raw.strip())
        correct = "ABCD"[idx - 1] if 1 <= idx <= 4 else "?"
    else:
        correct = str(raw).strip()

    return prompt, correct


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

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
    m = re.search(r"answer\s*(?:is|:)?\s*([ABCD])\b", upper)
    if m:
        return m.group(1)
    m = re.search(r"(?:therefore|correct(?:ly)?|option|choice)\s*[:\-]?\s*([ABCD])\b", upper)
    if m:
        return m.group(1)
    matches = re.findall(r"\b([ABCD])\b", upper)
    if matches:
        return matches[-1]
    return None


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def _build_input(tokenizer, prompt: str, device) -> dict:
    """Apply chat template if the tokenizer supports it, else plain text."""
    import torch
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return {"input_ids": ids.to(device)}
        except Exception:
            pass
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    return {"input_ids": ids.to(device)}


def _generate(model, tokenizer, inputs: dict, max_new_tokens: int = 10) -> str:
    import torch
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic, fastest
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Inference batch size (default 1 — safest for variable-length prompts)")
    args = parser.parse_args()

    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading tokenizer {MODEL_ID}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {MODEL_ID}…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto" if str(device) == "cuda" else None,
    )
    if str(device) != "cuda":
        model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  parameters: {n_params:.2f}B")

    print("Loading ai4bharat/MILU (Telugu)…")
    from datasets import load_dataset
    ds = load_dataset("ai4bharat/MILU", "Telugu", split="test", token=token)
    questions = list(ds)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"  {len(questions)} questions")

    results_dir = Path(args.experiment_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "milu_sarvam_raw.jsonl"
    acc_path = results_dir / "milu_sarvam_accuracy.json"
    yaml_path = results_dir / "benchmark_scores_milu_sarvam.yaml"

    done: dict[int, dict] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["idx"]] = rec
        print(f"  Resuming — {len(done)} already done.")

    checkpoint_fh = raw_path.open("a", encoding="utf-8")
    try:
        for idx, row in enumerate(questions):
            if idx in done:
                continue

            prompt, correct = _parse_row(dict(row))
            inputs = _build_input(tokenizer, prompt, device)
            raw_response = _generate(model, tokenizer, inputs)
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
                valid = [r for r in done.values() if r["predicted"] is not None]
                acc = sum(r["is_correct"] for r in valid) / len(valid) if valid else 0
                print(f"  [{idx+1}/{len(questions)}] running accuracy: {acc:.3f}")
    finally:
        checkpoint_fh.close()

    records = list(done.values())
    valid = [r for r in records if r["predicted"] is not None]
    unparseable = len(records) - len(valid)
    overall_acc = sum(r["is_correct"] for r in valid) / len(valid) if valid else 0.0

    by_subject: dict[str, list[bool]] = {}
    for r in valid:
        subj = r["subject"] or "unknown"
        by_subject.setdefault(subj, []).append(r["is_correct"])

    summary = {
        "model": MODEL_ID,
        "model_label": MODEL_LABEL,
        "benchmark": "MILU-Telugu",
        "split": "test",
        "evaluation_method": "zero-shot generative (greedy decoding)",
        "n_questions": len(records),
        "n_valid": len(valid),
        "n_unparseable": unparseable,
        "overall_accuracy": round(overall_acc, 4),
        "accuracy_pct": round(overall_acc * 100, 2),
        "paper_score_5shot": 28.57,
        "by_subject": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_subject.items())},
    }

    with acc_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    benchmark_yaml = {
        "benchmark": "MILU-Telugu (test, zero-shot)",
        "citation": "Verma et al. 2024 (arXiv:2411.02538)",
        "notes": (
            "sarvam-2b evaluated zero-shot generative (greedy decoding) on local model. "
            "Paper reports 28.57% at 5-shot for comparison. "
            "GPT-4o score (72.53%) is from MILU paper Table 3 (5-shot)."
        ),
        "accuracy_by_model": {
            "openai-gpt4o": 72.53,
            "sarvam-2b": round(overall_acc * 100, 2),
        },
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(benchmark_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\nDone.")
    print(f"  Overall accuracy : {overall_acc*100:.2f}%  ({len(valid)}/{len(records)} parseable)")
    print(f"  Paper score (5-shot): 28.57%")
    print(f"  Results          : {acc_path}")
    print(f"  Benchmark YAML   : {yaml_path}")


if __name__ == "__main__":
    main()
