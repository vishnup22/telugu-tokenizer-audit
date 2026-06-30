#!/usr/bin/env python3
"""Log-likelihood evaluation of Sarvam-2B on MILU Telugu."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_ID = "sarvamai/sarvam-2b-v0.5"
MODEL_LABEL = "sarvam-2b"

_PROMPT_TEMPLATE = """\\
Answer the following Telugu multiple-choice question by selecting the best option.

Question:
{question}

Choices:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Best answer:
"""


def _parse_row(row: dict) -> tuple[str, dict[str, str], str]:
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

    prompt_prefix = _PROMPT_TEMPLATE.format(
        question=question, opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d
    )
    options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}

    target_map = {"option1": "A", "option2": "B", "option3": "C", "option4": "D"}
    raw = row.get("target", row.get("answer", row.get("correct_option", row.get("Answer", ""))))
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

    return prompt_prefix, options, correct


def _encode_prompt(tokenizer, prompt: str, device):
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return ids.to(device)
        except Exception:
            pass
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def _score_option(model, tokenizer, prompt_ids, option_text: str, device) -> float:
    import torch

    option_ids = tokenizer(option_text, return_tensors="pt", add_special_tokens=False)
    option_ids = option_ids.input_ids.to(device)
    input_ids = torch.cat([prompt_ids, option_ids], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

    option_len = option_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    start = prompt_len - 1
    end = start + option_len
    target_ids = input_ids[:, prompt_len:]
    token_log_probs = log_probs[:, start:end, :].gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    return float(token_log_probs.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps")
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

    print(f"Loading tokenizer {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {MODEL_ID}...")
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

    print("Loading ai4bharat/MILU (Telugu)...")
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
        print(f"  Resuming - {len(done)} already done.")

    checkpoint_fh = raw_path.open("a", encoding="utf-8")
    try:
        for idx, row in enumerate(questions):
            if idx in done:
                continue

            prompt_prefix, options, correct = _parse_row(dict(row))
            prompt_ids = _encode_prompt(tokenizer, prompt_prefix, device)
            option_scores = {
                letter: _score_option(model, tokenizer, prompt_ids, text, device)
                for letter, text in options.items()
            }
            predicted = max(option_scores, key=option_scores.get)
            is_correct = predicted == correct

            rec = {
                "idx": idx,
                "subject": row.get("subject", row.get("topic", row.get("category", ""))),
                "domain": row.get("domain", row.get("category", "")),
                "correct": correct,
                "predicted": predicted,
                "option_scores": option_scores,
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
        "evaluation_method": "length-normalized log-likelihood over answer options",
        "n_questions": len(records),
        "n_valid": len(valid),
        "n_unparseable": unparseable,
        "overall_accuracy": round(overall_acc, 4),
        "accuracy_pct": round(overall_acc * 100, 2),
        "by_subject": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_subject.items())},
    }

    with acc_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    benchmark_yaml = {
        "benchmark": "MILU-Telugu (test)",
        "citation": "Verma et al. 2024 (arXiv:2411.02538)",
        "notes": (
            "sarvam-2b evaluated with length-normalized log-likelihood scoring on local model. "
            "GPT-4o score (72.53%) is from MILU paper Table 3 under a separate 5-shot protocol."
        ),
        "accuracy_by_model": {
            "openai-gpt4o": 72.53,
            "sarvam-2b": round(overall_acc * 100, 2),
        },
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(benchmark_yaml, f, allow_unicode=True, sort_keys=False)

    print("\nDone.")
    print(f"  Overall accuracy : {overall_acc*100:.2f}%")
    print("  Protocol         : length-normalized log-likelihood")
    print(f"  Results          : {acc_path}")
    print(f"  Benchmark YAML   : {yaml_path}")


if __name__ == "__main__":
    main()
