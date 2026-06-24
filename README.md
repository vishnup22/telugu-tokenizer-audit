# Telugu Tokenizer Fertility & Fairness Audit

Measures how efficiently major LLM tokenizers handle Telugu across three corpus registers, with a focus on native-script vs romanized fertility gap and morphological breakdown.

**Tokenizers evaluated:** GPT-4o, Claude Haiku, GPT-2, Sarvam-2B, Sarvam-105B, [dravidian-gpt2-telugu](https://huggingface.co/pulipakav-1/dravidian-gpt2-telugu)

## Key results

| Tokenizer | Fertility (tokens/word) | Script gap |
|---|---|---|
| dravidian-gpt2-telugu | 1.60 | 0.29 |
| Sarvam-105B | 2.71 | 0.68 |
| Sarvam-2B | 3.01 | 0.53 |
| GPT-4o | 3.25 | 0.82 |
| Claude Haiku | 7.73 | 1.56 |
| GPT-2 | 22.08 | 4.66 |

Script gap < 1.0 means native Telugu is tokenized *more* efficiently than romanized. GPT-2's gap of 4.66 means native script costs 4.66× more tokens than ITRANS romanization.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and add your API keys:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

## Running the pipeline

```bash
# 1. Collect corpus (~1000 sentences per register)
python scripts/collect_data.py --n-samples 1000
python scripts/01_build_corpus.py --config configs/default.yaml

# 2. Run fertility audit (all 6 tokenizers)
python scripts/02_run_tokenizer_audit.py --config configs/default.yaml --experiment-dir experiments/my_run

# 3. Minimal pair morphological breakdown
python scripts/03_run_minimal_pair_audit.py --config configs/default.yaml --experiment-dir experiments/my_run

# 4. Significance tests (Wilcoxon + bootstrap CI)
python scripts/06_run_significance_tests.py --config configs/default.yaml --experiment-dir experiments/my_run

# 5. Figures and tables
python scripts/05_make_figures_and_tables.py --experiment-dir experiments/my_run
```

Results are written to `experiments/my_run/results/`, `figures/`, and `tables/`.

## Corpus

Three registers collected automatically:
- `native_formal` — Telugu Wikipedia
- `native_informal` — Telugu social media (sentiment + hatespeech datasets)
- `romanized_informal` — ITRANS transliteration of native_informal (line-matched)

## Minimal pairs

`data/minimal_pairs/minimal_pairs.tsv` contains 62 native-speaker validated Telugu words across 9 morphological categories: base nouns, case suffixes (with/without sandhi), plural, honorific, compounds, borrowed words, and verb agglutination chains.

## Tests

```bash
pytest
```
