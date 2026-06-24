#!/usr/bin/env python3
"""Log-likelihood MCQ evaluation of dravidian-gpt2-telugu on MILU Telugu.

Base LMs can't follow instructions, so we use the standard log-likelihood
approach: for each question we score all four options as continuations of the
question context and pick the highest-scoring one.  The score is the mean
token log-probability of the option tokens given the question prefix
(length-normalised to avoid favouring longer options).

This runs entirely locally — no API calls, no cost.

Outputs (same experiment-dir/results/ as the Claude run):
  milu_dravidian_raw.jsonl          -- one JSON line per question (checkpoint)
  milu_dravidian_accuracy.json      -- overall + per-subject breakdown
  benchmark_scores_milu_dravidian.yaml  -- for 04_correlate_with_benchmarks.py

Usage:
    python scripts/08_run_milu_eval_dravidian.py \\
        --experiment-dir experiments/2026-06-23_v3_telugu_gpt2

    # Quick smoke-test on 20 questions:
    python scripts/08_run_milu_eval_dravidian.py \\
        --experiment-dir experiments/2026-06-23_v3_telugu_gpt2 \\
        --max-questions 20

Requires: transformers, torch, sentencepiece, datasets, huggingface_hub
"""

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

MODEL_ID = "pulipakav-1/dravidian-gpt2-telugu"
MODEL_LABEL = "telugu-gpt2"


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def _load_sp_tokenizer(token: str | None):
    """Load SentencePiece tokenizer from the HF repo."""
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(MODEL_ID, "tokenizer.model", token=token)
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def _encode_ids(sp, text: str) -> list[int]:
    return sp.encode(text, out_type=int)


# ---------------------------------------------------------------------------
# Log-likelihood scoring
# ---------------------------------------------------------------------------

def _score_option(
    model,
    input_ids,          # full sequence: [question_ids + option_ids]
    option_len: int,
    device,
) -> float:
    """Return mean log-prob of the last `option_len` tokens given the prefix."""
    import torch
    ids = input_ids.to(device)
    with torch.no_grad():
        out = model(ids, labels=ids)
        # out.logits: (1, seq_len, vocab)
        logits = out.logits[0]          # (seq_len, vocab)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Position i predicts token i+1; option starts at position (seq_len - option_len)
    seq_len = ids.shape[1]
    start = seq_len - option_len - 1   # logit at this position predicts first option token
    # Gather log-probs for each option token
    option_ids = ids[0, seq_len - option_len:]   # the actual option token ids
    option_logits_positions = list(range(start, start + option_len))
    score = sum(
        log_probs[pos, tok_id].item()
        for pos, tok_id in zip(option_logits_positions, option_ids)
    )
    return score / option_len   # length-normalised


# ---------------------------------------------------------------------------
# Build prompt components
# ---------------------------------------------------------------------------

def _parse_row(row: dict) -> tuple[str, list[str], str]:
    """Return (question_text, [opt_a, opt_b, opt_c, opt_d], correct_letter)."""
    question = (
        row.get("question") or row.get("question_text") or row.get("Question") or ""
    ).strip()

    if "option1" in row:
        opts = [str(row.get(f"option{i}", "")) for i in range(1, 5)]
    elif "option_1" in row:
        opts = [str(row.get(f"option_{i}", "")) for i in range(1, 5)]
    elif "options" in row and isinstance(row["options"], list):
        opts = (list(row["options"]) + [""] * 4)[:4]
    else:
        opts = [str(row.get(k, "")) for k in ("option_a", "option_b", "option_c", "option_d")]

    _tmap = {"option1": "A", "option2": "B", "option3": "C", "option4": "D"}
    raw = row.get("target", row.get("answer", row.get("correct_option", "")))
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

    return question, opts, correct


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
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu / mps — auto-detected if omitted",
    )
    args = parser.parse_args()

    import os
    import torch
    from transformers import AutoModelForCausalLM

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # load tokenizer
    print(f"Loading SentencePiece tokenizer from {MODEL_ID}…")
    sp = _load_sp_tokenizer(token)
    print(f"  vocab size: {sp.vocab_size()}")

    # load model
    print(f"Loading model {MODEL_ID}…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        trust_remote_code=False,
    )
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  parameters: {n_params:.0f}M")

    # load MILU
    print("Loading ai4bharat/MILU (Telugu)…")
    from datasets import load_dataset
    ds = load_dataset("ai4bharat/MILU", "Telugu", split="test", token=token)
    questions = list(ds)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"  {len(questions)} questions")

    # output paths
    results_dir = Path(args.experiment_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "milu_dravidian_raw.jsonl"
    acc_path = results_dir / "milu_dravidian_accuracy.json"
    yaml_path = results_dir / "benchmark_scores_milu_dravidian.yaml"

    # resume
    done: dict[int, dict] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["idx"]] = rec
        print(f"  Resuming — {len(done)} already done.")

    import torch

    checkpoint_fh = raw_path.open("a", encoding="utf-8")
    try:
        for idx, row in enumerate(questions):
            if idx in done:
                continue

            question, opts, correct = _parse_row(dict(row))

            # Encode question prefix once
            q_text = f"Question: {question}\nAnswer:"
            q_ids = _encode_ids(sp, q_text)

            scores: list[float] = []
            letters = ["A", "B", "C", "D"]
            for letter, opt_text in zip(letters, opts):
                opt_str = f" {letter}. {opt_text}"
                opt_ids = _encode_ids(sp, opt_str)
                if not opt_ids:
                    scores.append(-math.inf)
                    continue
                full_ids = torch.tensor([q_ids + opt_ids], dtype=torch.long)
                s = _score_option(model, full_ids, len(opt_ids), device)
                scores.append(s)

            predicted = letters[scores.index(max(scores))]
            is_correct = predicted == correct

            rec = {
                "idx": idx,
                "subject": row.get("subject", row.get("topic", row.get("category", ""))),
                "domain": row.get("domain", row.get("category", "")),
                "correct": correct,
                "predicted": predicted,
                "scores": [round(s, 6) for s in scores],
                "is_correct": is_correct,
            }
            done[idx] = rec
            checkpoint_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            checkpoint_fh.flush()

            if (idx + 1) % 100 == 0:
                acc = sum(r["is_correct"] for r in done.values()) / len(done)
                print(f"  [{idx+1}/{len(questions)}] running accuracy: {acc:.3f}")
    finally:
        checkpoint_fh.close()

    # aggregate
    records = list(done.values())
    overall_acc = sum(r["is_correct"] for r in records) / len(records) if records else 0.0

    by_subject: dict[str, list[bool]] = {}
    for r in records:
        subj = r["subject"] or "unknown"
        by_subject.setdefault(subj, []).append(r["is_correct"])

    summary = {
        "model": MODEL_ID,
        "model_label": MODEL_LABEL,
        "benchmark": "MILU-Telugu",
        "split": "test",
        "evaluation_method": "log-likelihood (length-normalised)",
        "n_questions": len(records),
        "overall_accuracy": round(overall_acc, 4),
        "accuracy_pct": round(overall_acc * 100, 2),
        "chance_baseline": 25.0,
        "by_subject": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_subject.items())},
    }

    with acc_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    benchmark_yaml = {
        "benchmark": "MILU-Telugu (test, log-likelihood)",
        "citation": "Verma et al. 2024 (arXiv:2411.02538)",
        "notes": (
            "dravidian-gpt2-telugu evaluated via log-likelihood (length-normalised) — "
            "base LM, no instruction following. Chance baseline = 25.0%."
        ),
        "accuracy_by_model": {
            MODEL_LABEL: round(overall_acc * 100, 2),
        },
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(benchmark_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\nDone.")
    print(f"  Overall accuracy : {overall_acc*100:.2f}%  (chance = 25.0%)")
    print(f"  Results          : {acc_path}")
    print(f"  Benchmark YAML   : {yaml_path}")


if __name__ == "__main__":
    main()
