                      
"""Log-likelihood evaluation of Sarvam-2B on MILU for a supported language."""

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

from telugu_audit.benchmarks.milu import (
    build_milu_output_paths,
    milu_benchmark_name,
    milu_config_name,
    milu_language_slug,
    parse_milu_row,
    row_subject,
)

MODEL_ID = "sarvamai/sarvam-2b-v0.5"
MODEL_LABEL = "sarvam-2b"
_PROMPT_TEMPLATE = """\
Answer the following multiple-choice question by selecting the best option.

Question:
{question}

Choices:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Best answer:
"""


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

    option_ids = tokenizer(option_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, option_ids], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

    option_len = option_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    start = prompt_len - 1
    end = start + option_len
    target_ids = input_ids[:, prompt_len:]
    token_log_probs = log_probs[:, start:end, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--language", default="telugu", help="Supported: telugu, hindi")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps")
    args = parser.parse_args()

    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    language_label = milu_config_name(args.language)
    language_slug = milu_language_slug(args.language)

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
    print(f"  parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    print(f"Loading ai4bharat/MILU ({language_label})...")
    ds = load_dataset("ai4bharat/MILU", language_label, split="test", token=token)
    questions = list(ds)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"  {len(questions)} questions")

    results_dir = Path(args.experiment_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = build_milu_output_paths(results_dir, "sarvam", args.language)
    raw_path = paths["raw"]
    acc_path = paths["accuracy"]
    yaml_path = paths["benchmark_yaml"]

    done: dict[int, dict] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                done[record["idx"]] = record
        print(f"  Resuming - {len(done)} already done.")

    checkpoint_handle = raw_path.open("a", encoding="utf-8")
    try:
        for idx, row in enumerate(questions):
            if idx in done:
                continue

            question, options, correct = parse_milu_row(dict(row))
            prompt_prefix = _PROMPT_TEMPLATE.format(
                question=question,
                opt_a=options["A"],
                opt_b=options["B"],
                opt_c=options["C"],
                opt_d=options["D"],
            )
            prompt_ids = _encode_prompt(tokenizer, prompt_prefix, device)
            option_scores = {letter: _score_option(model, tokenizer, prompt_ids, text, device) for letter, text in options.items()}
            predicted = max(option_scores, key=option_scores.get)
            is_correct = predicted == correct

            record = {
                "idx": idx,
                "language": language_slug,
                "subject": row_subject(row),
                "correct": correct,
                "predicted": predicted,
                "option_scores": option_scores,
                "is_correct": is_correct,
            }
            done[idx] = record
            checkpoint_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint_handle.flush()

            if (idx + 1) % 100 == 0:
                valid = [item for item in done.values() if item["predicted"] is not None]
                accuracy = sum(item["is_correct"] for item in valid) / len(valid) if valid else 0.0
                print(f"  [{idx + 1}/{len(questions)}] running accuracy: {accuracy:.3f}")
    finally:
        checkpoint_handle.close()

    records = list(done.values())
    valid = [record for record in records if record["predicted"] is not None]
    unparseable = len(records) - len(valid)
    overall_accuracy = sum(record["is_correct"] for record in valid) / len(valid) if valid else 0.0

    by_subject: dict[str, list[bool]] = {}
    for record in valid:
        by_subject.setdefault(record["subject"], []).append(record["is_correct"])

    summary = {
        "model": MODEL_ID,
        "model_label": MODEL_LABEL,
        "benchmark": milu_benchmark_name(args.language),
        "language": language_slug,
        "split": "test",
        "evaluation_method": "length-normalized log-likelihood over answer options",
        "n_questions": len(records),
        "n_valid": len(valid),
        "n_unparseable": unparseable,
        "overall_accuracy": round(overall_accuracy, 4),
        "accuracy_pct": round(overall_accuracy * 100, 2),
        "by_subject": {key: round(sum(values) / len(values), 4) for key, values in sorted(by_subject.items())},
    }

    with acc_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    benchmark_yaml = {
        "benchmark": f"{milu_benchmark_name(args.language)} (test)",
        "citation": "Verma et al. 2024 (arXiv:2411.02538)",
        "notes": (
            "sarvam-2b evaluated with length-normalized log-likelihood scoring on the local model. "
            "Published comparison points should be added per language before correlation."
        ),
        "accuracy_by_model": {
            MODEL_LABEL: round(overall_accuracy * 100, 2),
        },
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.dump(benchmark_yaml, handle, allow_unicode=True, sort_keys=False)

    print("\nDone.")
    print(f"  Overall accuracy : {overall_accuracy * 100:.2f}%")
    print("  Protocol         : length-normalized log-likelihood")
    print(f"  Results          : {acc_path}")
    print(f"  Benchmark YAML   : {yaml_path}")


if __name__ == "__main__":
    main()
