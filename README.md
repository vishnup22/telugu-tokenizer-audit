# Telugu Tokenizer Fertility and Fairness Audit

A reproducible empirical audit of how multiple LLM tokenizers handle Telugu across formal, informal, romanized, and English baseline registers. The repo measures fertility, script fairness, morphological sensitivity, and the pricing/context implications of tokenizer design.

## Current status

The main local revision items are now implemented in the codebase and rerun on the latest full experiment:
- dual-denominator fertility outputs with `whitespace` and `indicnlp`
- two-sided Wilcoxon signed-rank testing
- rank-biserial effect sizes
- BCa bootstrap median confidence intervals
- minimal-pair variance summaries and Kruskal-Wallis tests
- `xlm-r`, `indicbert`, and `llama-3` tokenizer runs
- direct `english_wiki` baseline collection and integration
- Sarvam log-likelihood MILU evaluation script

Remaining incomplete items are mainly final paper polish and long-running benchmark jobs rather than core audit plumbing.

## Latest full run

Primary current run:
- `experiments/2026-06-30_full_with_llama_english_r1`

Tokenizers present in that run:
- `openai-gpt4o`
- `claude`
- `hf-gpt2`
- `sarvam-2b`
- `sarvam-105b`
- `telugu-gpt2`
- `xlm-r`
- `indicbert`
- `llama-3`

Registers present in that run:
- `native_formal`
- `native_informal`
- `romanized_informal`
- `english_wiki`

## Main findings from the latest run

Native informal fertility, lower is better:

| Tokenizer | Tokens/word |
|---|---:|
| `telugu-gpt2` | 1.60 |
| `xlm-r` | 2.41 |
| `sarvam-105b` | 2.71 |
| `indicbert` | 2.93 |
| `sarvam-2b` | 3.01 |
| `openai-gpt4o` | 3.25 |
| `claude` | 7.73 |
| `llama-3` | 14.15 |
| `hf-gpt2` | 22.08 |

Script fairness ratio `native / romanized`, where values above 1 mean native Telugu is costlier:

| Tokenizer | Ratio | Interpretation |
|---|---:|---|
| `hf-gpt2` | 4.66 | native Telugu much costlier |
| `llama-3` | 3.29 | native Telugu much costlier |
| `claude` | 1.56 | native Telugu costlier |
| `indicbert` | 0.89 | romanized slightly costlier |
| `openai-gpt4o` | 0.82 | romanized costlier |
| `sarvam-105b` | 0.68 | romanized costlier |
| `xlm-r` | 0.61 | romanized costlier |
| `sarvam-2b` | 0.53 | romanized costlier |
| `telugu-gpt2` | 0.29 | romanized much costlier |

Measured Telugu/English multiplier on `native_informal / english_wiki`:

| Tokenizer | Telugu | English | Telugu/English |
|---|---:|---:|---:|
| `claude` | 7.73 | 1.46 | 5.31x |
| `openai-gpt4o` | 3.25 | 1.33 | 2.44x |
| `llama-3` | 14.15 | 1.35 | 10.49x |
| `hf-gpt2` | 22.08 | 1.34 | 16.49x |

## Corpus

Current local corpus registers:
- `native_formal`: Telugu Wikipedia-derived formal text
- `native_informal`: social-media style Telugu text
- `romanized_informal`: line-aligned ITRANS transliteration of `native_informal`
- `english_wiki`: English Wikipedia baseline collected with the same sentence-length filter

The current reported romanized corpus is still a controlled ITRANS baseline, not organic Tenglish.

## Minimal pairs

`data/minimal_pairs/minimal_pairs.tsv` contains 62 Telugu forms across 9 morphological categories.

Summary outputs for the latest run:
- `experiments/2026-06-30_full_with_llama_english_r1/results/minimal_pair_summary.csv`
- `experiments/2026-06-30_full_with_llama_english_r1/results/minimal_pair_kruskal_wallis.csv`

## Running the pipeline

```bash
python scripts/collect_data.py --n-samples 1000 --include-english
python scripts/01_build_corpus.py --config configs/default.yaml
python scripts/02_run_tokenizer_audit.py --config configs/default.yaml --run_tag myrun
python scripts/03_run_minimal_pair_audit.py --config configs/default.yaml --experiment-dir experiments/YYYY-MM-DD_myrun
python scripts/06_run_significance_tests.py --config configs/default.yaml --experiment-dir experiments/YYYY-MM-DD_myrun
python scripts/05_make_figures_and_tables.py --experiment-dir experiments/YYYY-MM-DD_myrun
```

## Benchmarks

MILU scripts exist for:
- Claude generative evaluation
- Telugu GPT-2 log-likelihood evaluation
- Sarvam-2B log-likelihood evaluation

Long-running full MILU jobs for the latest run are currently writing logs under:
- `experiments/2026-06-30_full_with_llama_english_r1/results/`

## Tests

```bash
python -m pytest
```

## License

MIT
