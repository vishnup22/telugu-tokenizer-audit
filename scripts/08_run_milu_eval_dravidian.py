                      
"""Log-likelihood MCQ evaluation of dravidian-gpt2-telugu on MILU."""

from __future__ import annotations

import argparse
import json
import math
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

MODEL_ID = "pulipakav-1/dravidian-gpt2-telugu"
MODEL_LABEL = "telugu-gpt2"
MODEL_LANGUAGES = {"telugu"}


def _load_sp_tokenizer(token: str | None):
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_ID, "tokenizer.model", token=token)
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def _encode_ids(sp, text: str) -> list[int]:
    return sp.encode(text, out_type=int)


def _score_option(model, input_ids, option_len: int, device) -> float:
    import torch

    ids = input_ids.to(device)
    with torch.no_grad():
        output = model(ids, labels=ids)
        logits = output.logits[0]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    sequence_length = ids.shape[1]
    start = sequence_length - option_len - 1
    option_ids = ids[0, sequence_length - option_len:]
    positions = range(start, start + option_len)
    score = sum(log_probs[position, token_id].item() for position, token_id in zip(positions, option_ids))
    return score / option_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--language", default="telugu", help="Supported by default: telugu")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps")
    parser.add_argument(
        "--allow-cross-lingual",
        action="store_true",
        help="Allow evaluating this Telugu-specific model on a non-Telugu MILU split.",
    )
    args = parser.parse_args()

    language_slug = milu_language_slug(args.language)
    if language_slug not in MODEL_LANGUAGES and not args.allow_cross_lingual:
        supported = ", ".join(sorted(MODEL_LANGUAGES))
        raise SystemExit(
            f"{MODEL_LABEL} is configured only for {supported}. "
            "Pass --allow-cross-lingual if you intentionally want a cross-lingual stress test."
        )

    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    language_label = milu_config_name(args.language)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading SentencePiece tokenizer from {MODEL_ID}...")
    sp = _load_sp_tokenizer(token)
    print(f"  vocab size: {sp.vocab_size()}")

    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=token, trust_remote_code=False)
    model.eval()
    model.to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    print(f"Loading ai4bharat/MILU ({language_label})...")
    ds = load_dataset("ai4bharat/MILU", language_label, split="test", token=token)
    questions = list(ds)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"  {len(questions)} questions")

    results_dir = Path(args.experiment_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = build_milu_output_paths(results_dir, "dravidian", args.language)
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
            question_ids = _encode_ids(sp, f"Question: {question}\nAnswer:")

            scores: list[float] = []
            letters = ["A", "B", "C", "D"]
            for letter in letters:
                option_text = f" {letter}. {options[letter]}"
                option_ids = _encode_ids(sp, option_text)
                if not option_ids:
                    scores.append(-math.inf)
                    continue
                full_ids = torch.tensor([question_ids + option_ids], dtype=torch.long)
                scores.append(_score_option(model, full_ids, len(option_ids), device))

            predicted = letters[scores.index(max(scores))]
            is_correct = predicted == correct
            record = {
                "idx": idx,
                "language": language_slug,
                "subject": row_subject(row),
                "correct": correct,
                "predicted": predicted,
                "scores": [round(score, 6) for score in scores],
                "is_correct": is_correct,
            }
            done[idx] = record
            checkpoint_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint_handle.flush()

            if (idx + 1) % 100 == 0:
                accuracy = sum(item["is_correct"] for item in done.values()) / len(done)
                print(f"  [{idx + 1}/{len(questions)}] running accuracy: {accuracy:.3f}")
    finally:
        checkpoint_handle.close()

    records = list(done.values())
    overall_accuracy = sum(record["is_correct"] for record in records) / len(records) if records else 0.0

    by_subject: dict[str, list[bool]] = {}
    for record in records:
        by_subject.setdefault(record["subject"], []).append(record["is_correct"])

    summary = {
        "model": MODEL_ID,
        "model_label": MODEL_LABEL,
        "benchmark": milu_benchmark_name(args.language),
        "language": language_slug,
        "split": "test",
        "evaluation_method": "log-likelihood (length-normalized)",
        "n_questions": len(records),
        "overall_accuracy": round(overall_accuracy, 4),
        "accuracy_pct": round(overall_accuracy * 100, 2),
        "chance_baseline": 25.0,
        "by_subject": {key: round(sum(values) / len(values), 4) for key, values in sorted(by_subject.items())},
    }

    with acc_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    benchmark_yaml = {
        "benchmark": f"{milu_benchmark_name(args.language)} (test, log-likelihood)",
        "citation": "Verma et al. 2024 (arXiv:2411.02538)",
        "notes": (
            "dravidian-gpt2-telugu evaluated via length-normalized log-likelihood. "
            "If run outside Telugu, interpret as a cross-lingual stress test rather than an in-language benchmark."
        ),
        "accuracy_by_model": {
            MODEL_LABEL: round(overall_accuracy * 100, 2),
        },
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.dump(benchmark_yaml, handle, allow_unicode=True, sort_keys=False)

    print("\nDone.")
    print(f"  Overall accuracy : {overall_accuracy * 100:.2f}%  (chance = 25.0%)")
    print(f"  Results          : {acc_path}")
    print(f"  Benchmark YAML   : {yaml_path}")


if __name__ == "__main__":
    main()
