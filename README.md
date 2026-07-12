# Telugu Tokenizer Fertility and Fairness Audit

A reproducible empirical audit of how multiple LLM tokenizers handle Telugu across formal, informal, romanized, Tenglish, and English baseline registers. The repo measures fertility, script fairness, morphological sensitivity, and the pricing/context implications of tokenizer design.

## Current Status

The revision work is now implemented in code and reflected in the latest experiment outputs:
- dual-denominator fertility outputs with `whitespace` and `indicnlp`
- two-sided Wilcoxon signed-rank testing
- rank-biserial effect sizes
- BCa bootstrap median confidence intervals
- minimal-pair variance summaries and Kruskal-Wallis tests
- `xlm-r`, `indicbert`, and `llama-3` tokenizer runs
- direct `english_wiki` baseline collection and integration
- `tenglish_informal` sourced from a 3,300-comment YouTube export and filtered down to 249 lines
- Sarvam and Dravidian MILU log-likelihood evaluation scripts
- Claude MILU completed locally with the final benchmark result
- Sarvam and Dravidian MILU completed on Colab GPU and pushed back to the repo

The Tenglish register is now populated and analyzed. The tracked public-source manifest is `tenglish_sources.md`.

## Latest Artifacts

Main audit run:
- `experiments/2026-06-30_full_with_llama_english_r1`

Tenglish rerun:
- `experiments/2026-07-01_tenglish_r1`

GPU benchmark run:
- `experiments/2026-06-30_full_with_llama_english_r1_gpu`

Loaded tokenizers in the main run:
- `openai-gpt4o`
- `claude`
- `hf-gpt2`
- `sarvam-2b`
- `sarvam-105b`
- `telugu-gpt2`
- `xlm-r`
- `indicbert`
- `llama-3`

## Main Findings

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
| `hf-gpt2` | 22.08 | 1.34 | 16.49x |
| `llama-3` | 14.15 | 1.35 | 10.49x |
| `claude` | 7.73 | 1.46 | 5.31x |
| `openai-gpt4o` | 3.25 | 1.33 | 2.44x |
| `sarvam-105b` | 2.71 | 1.35 | 2.01x |
| `indicbert` | 2.93 | 1.52 | 1.92x |
| `sarvam-2b` | 3.01 | 1.60 | 1.89x |
| `xlm-r` | 2.41 | 1.52 | 1.59x |
| `telugu-gpt2` | 1.60 | 2.65 | 0.60x |

Tenglish vs native informal on the 249-line YouTube sample:

| Tokenizer | Native informal | Tenglish | Native/Tenglish |
|---|---:|---:|---:|
| `openai-gpt4o` | 3.32 | 1.72 | 1.93x |
| `claude` | 7.81 | 2.31 | 3.37x |
| `hf-gpt2` | 22.28 | 2.17 | 10.27x |
| `sarvam-2b` | 3.06 | 2.73 | 1.03x |
| `sarvam-105b` | 2.76 | 1.54 | 1.75x |
| `telugu-gpt2` | 1.63 | 2.64 | 0.62x |
| `xlm-r` | 2.46 | 1.66 | 1.48x |
| `indicbert` | 2.97 | 1.53 | 1.95x |
| `llama-3` | 14.30 | 1.88 | 7.17x |

MILU Telugu accuracy, final run values:

| Model | Method | Accuracy |
|---|---|---:|
| `openai-gpt4o` | paper-reported 5-shot | 72.53% |
| `claude` | local zero-shot generative | 60.92% |
| `sarvam-2b` | log-likelihood | 28.19% |
| `telugu-gpt2` | log-likelihood | 26.97% |
| chance baseline | - | 25.00% |

## Corpus

Current local corpus registers:
- `native_formal`: Telugu Wikipedia-derived formal text
- `native_informal`: social-media style Telugu text
- `romanized_informal`: line-aligned ITRANS transliteration of `native_informal`
- `english_wiki`: English Wikipedia baseline collected with the same sentence-length filter
- `tenglish_informal`: 249-line organic Latin-script Telugu sample filtered from public YouTube comments

## Minimal Pairs

`data/minimal_pairs/minimal_pairs.tsv` contains 62 Telugu forms across 9 morphological categories:
- `base_noun`
- `case_suffix`
- `case_suffix_with_sandhi`
- `plural_suffix`
- `honorific_suffix`
- `compound_with_sandhi`
- `borrowed_base`
- `borrowed_plus_case_suffix`
- `verb_agglutination_chain`

Latest outputs:
- `experiments/2026-07-01_tenglish_r1/results/minimal_pair_summary.csv`
- `experiments/2026-07-01_tenglish_r1/results/minimal_pair_kruskal_wallis.csv`

## Correlation

The updated fertility-versus-accuracy correlation for the available benchmark set is written to:
- `experiments/2026-07-01_tenglish_r1/results/fertility_accuracy_correlation.json`

The set is still small, so this remains descriptive rather than definitive.

## Reproduction

```bash
python scripts/collect_data.py --n-samples 1000 --include-english
python scripts/01_build_corpus.py --config configs/default.yaml
python scripts/02_run_tokenizer_audit.py --config configs/default.yaml --run_tag myrun
python scripts/03_run_minimal_pair_audit.py --config configs/default.yaml --experiment-dir experiments/YYYY-MM-DD_myrun
python scripts/06_run_significance_tests.py --config configs/default.yaml --experiment-dir experiments/YYYY-MM-DD_myrun
python scripts/04_correlate_with_benchmarks.py --experiment-dir experiments/YYYY-MM-DD_myrun --benchmark-scores experiments/YYYY-MM-DD_myrun/results/benchmark_scores_milu.yaml
python scripts/05_make_figures_and_tables.py --experiment-dir experiments/YYYY-MM-DD_myrun
```

## Tests

```bash
python -m pytest
```

## License

MIT
