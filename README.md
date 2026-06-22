# Telugu Tokenizer Fertility & Fairness Audit

Reproducible audit of how major LLM tokenizers handle Telugu — fertility by
register, native-vs-romanized script fairness, minimal-pair morphological
breakdown, cost/context translation, and benchmark correlation.

See [PLAN.md](PLAN.md) for the full build specification.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and fill in API keys for live tokenizer runs.

## Quick test (no API keys)

```bash
pytest
```

## Pipeline

```bash
make corpus    # data/raw -> data/processed
make audit     # fertility + minimal-pair audits
make figures   # tables and plots from latest experiment
```

Each run writes a timestamped folder under `experiments/` with config snapshot,
`run_metadata.json`, and result CSVs.

## Human gates (before real-data runs)

Real corpus files, validated minimal pairs, scraping compliance, gated model
access, benchmark accuracy numbers, and current API pricing must be supplied
by a human before reporting paper numbers. See PLAN.md Section 7.
