# Telugu Tokenizer Fertility & Fairness Audit

A reproducible empirical audit measuring how six LLM tokenizers handle Telugu across three corpus registers. The study quantifies the **script fairness gap** — the token cost penalty imposed on native Telugu script relative to romanized Telugu — and provides a morphological breakdown via 62 validated minimal pairs.

## Findings

Fertility = mean tokens per whitespace-delimited word on the `native_informal` register.
Script gap = fertility(native) / fertility(romanized); values below 1.0 indicate native script is *cheaper*.

| Tokenizer | Fertility | Script gap |
|---|:---:|:---:|
| dravidian-gpt2-telugu (32k BPE, Telugu) | **1.60** | **0.29** |
| Sarvam-105B | 2.71 | 0.68 |
| Sarvam-2B | 3.01 | 0.53 |
| GPT-4o | 3.25 | 0.82 |
| Claude Haiku | 7.73 | 1.56 |
| GPT-2 (English, baseline) | 22.08 | 4.66 |

A Telugu-specialized 32k BPE tokenizer achieves 2× lower fertility than GPT-4o and fully eliminates the script bias (gap = 0.29). GPT-2's gap of 4.66 represents the worst-case penalty: native Telugu costs 4.66× more tokens than ITRANS romanization of the same text.

Wilcoxon signed-rank tests (one-sided, H₁: native > romanized, n = 1000 matched pairs):
- GPT-2, Claude: p < 0.0001 — native script significantly costlier
- Sarvam-2B, Sarvam-105B, dravidian-gpt2-telugu: p = 1.0 — romanized is costlier (gap inverted)
- GPT-4o: p = 1.0, median gap = −0.73 (marginal romanized advantage)

## Corpus

| Register | Source | N |
|---|---|---|
| `native_formal` | Telugu Wikipedia (`vengi-ai/telugu-wikipedia-clean`) | 789 |
| `native_informal` | Telugu sentiment + hatespeech datasets (HuggingFace) | 1,000 |
| `romanized_informal` | ITRANS transliteration of `native_informal` | 1,000 |

The romanized corpus is produced by systematic ITRANS transliteration (via `indic-transliteration`) of the native informal corpus, guaranteeing line-level alignment for paired significance testing.

## Minimal pairs

`data/minimal_pairs/minimal_pairs.tsv` — 62 native-speaker validated Telugu word forms across 9 morphological categories:

`base_noun` · `case_suffix` · `case_suffix_with_sandhi` · `plural_suffix` · `honorific_suffix` · `compound_with_sandhi` · `borrowed_base` · `borrowed_plus_case_suffix` · `verb_agglutination_chain`

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env          # add ANTHROPIC_API_KEY and OPENAI_API_KEY
```

## Reproducing the results

```bash
# Collect corpus
python scripts/collect_data.py --n-samples 1000
python scripts/01_build_corpus.py --config configs/default.yaml

# Run all tokenizers
python scripts/02_run_tokenizer_audit.py --config configs/default.yaml --experiment-dir experiments/run1

# Morphological breakdown
python scripts/03_run_minimal_pair_audit.py --config configs/default.yaml --experiment-dir experiments/run1

# Significance tests
python scripts/06_run_significance_tests.py --config configs/default.yaml --experiment-dir experiments/run1

# Export figures and tables
python scripts/05_make_figures_and_tables.py --experiment-dir experiments/run1
```

Results are written to `experiments/run1/results/`, `figures/`, and `tables/`.

Or using make:

```bash
make corpus && make audit EXPERIMENT_DIR=experiments/run1
```

## Repository layout

```
configs/          tokenizer sets, pricing, default config
data/
  minimal_pairs/  validated TSV of 62 word forms (9 morph types)
  raw/            collected corpus (gitignored)
  processed/      pipeline-ready corpus (gitignored)
experiments/      timestamped run outputs (results gitignored)
scripts/          numbered pipeline scripts
telugu_audit/     library: tokenizer adapters, corpus loaders, analysis
tests/            unit tests
```

## Tests

```bash
pytest
```

## License

MIT
