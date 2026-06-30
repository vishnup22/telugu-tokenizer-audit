# Telugu Tokenizer Fertility and Fairness Audit - Final Local Run Snapshot

**Primary latest run:** `experiments/2026-06-30_full_with_llama_english_r1/`  
**Corpus:** 789 `native_formal` + 1,000 `native_informal` + 1,000 `romanized_informal` + 1,000 `english_wiki`  
**Primary word-count method:** `whitespace`  
**Sensitivity method also run:** `indicnlp`  
**Loaded tokenizers:** `openai-gpt4o`, `claude`, `hf-gpt2`, `sarvam-2b`, `sarvam-105b`, `telugu-gpt2`, `xlm-r`, `indicbert`, `llama-3`

---

## 1. Fertility by Register

| Tokenizer | Native Formal | Native Informal | Romanized Informal | English Wiki |
|---|---:|---:|---:|---:|
| `openai-gpt4o` | 3.28 | 3.25 | 3.98 | 1.33 |
| `claude` | 6.91 | 7.73 | 4.95 | 1.46 |
| `hf-gpt2` | 19.99 | 22.08 | 4.74 | 1.34 |
| `sarvam-2b` | 2.52 | 3.01 | 5.67 | 1.60 |
| `sarvam-105b` | 2.68 | 2.71 | 4.01 | 1.35 |
| `telugu-gpt2` | 1.68 | 1.60 | 5.49 | 2.65 |
| `xlm-r` | 2.50 | 2.41 | 3.93 | 1.52 |
| `indicbert` | 2.86 | 2.93 | 3.31 | 1.52 |
| `llama-3` | 12.96 | 14.15 | 4.30 | 1.35 |

Key findings:
- `telugu-gpt2` remains the most efficient tokenizer on native Telugu.
- `xlm-r`, `indicbert`, and both Sarvam tokenizers are all far more efficient than Claude, Llama 3, and legacy GPT-2 on native Telugu.
- `llama-3` behaves much closer to the English-centric baselines than to the Indic-specialized tokenizers.

---

## 2. Measured Telugu/English Multiplier

Using `native_informal / english_wiki` from the same run:

| Tokenizer | Multiplier |
|---|---:|
| `claude` | 5.31x |
| `openai-gpt4o` | 2.44x |
| `sarvam-105b` | 2.01x |
| `sarvam-2b` | 1.89x |
| `xlm-r` | 1.59x |
| `indicbert` | 1.92x |
| `llama-3` | 10.49x |
| `hf-gpt2` | 16.49x |
| `telugu-gpt2` | 0.60x |

The direct English baseline is now measured rather than borrowed.

---

## 3. Script Fairness Gap

Script fairness ratio = `native_informal / romanized_informal`.

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

This still shows a clean split:
- English-heavy tokenizers can strongly penalize native Telugu script.
- Indic-trained or multilingual tokenizers with stronger Telugu coverage make native Telugu cheaper than the ITRANS baseline.

---

## 4. Revised Significance Outputs

The upgraded pipeline now reports:
- two-sided Wilcoxon signed-rank tests
- median-gap direction
- rank-biserial correlation
- bootstrap 95% median-gap confidence intervals

Selected results from `script_gap_significance.json`:

| Tokenizer | Median gap | 95% CI | Rank-biserial | Direction |
|---|---:|---|---:|---|
| `openai-gpt4o` | -0.730 | [-0.763, -0.684] | -0.913 | romanized costlier |
| `claude` | 2.750 | [2.714, 2.809] | 1.000 | native costlier |
| `hf-gpt2` | 17.222 | [17.000, 17.417] | 1.000 | native costlier |
| `sarvam-2b` | -2.655 | [-2.700, -2.583] | -1.000 | romanized costlier |
| `sarvam-105b` | -1.300 | [-1.333, -1.250] | -0.999 | romanized costlier |
| `telugu-gpt2` | -3.877 | [-4.000, -3.833] | -1.000 | romanized costlier |
| `xlm-r` | -1.550 | [-1.600, -1.500] | -0.997 | romanized costlier |
| `indicbert` | -0.400 | [-0.400, -0.364] | -0.927 | romanized costlier |
| `llama-3` | 9.793 | [9.636, 9.922] | 1.000 | native costlier |

---

## 5. Morphological Summary

Representative high-cost categories remain:
- `honorific_suffix`
- `verb_agglutination_chain`
- `borrowed_plus_case_suffix`

Representative values from `minimal_pair_summary.csv`:
- `hf-gpt2` on `honorific_suffix`: `35.50 +/- 6.95`
- `llama-3` on `honorific_suffix`: `23.67 +/- 4.63`
- `claude` on `honorific_suffix`: `11.67 +/- 3.44`
- `indicbert` on `honorific_suffix`: `4.00 +/- 1.26`
- `telugu-gpt2` stays near 1-2 tokens across most categories

Kruskal-Wallis by tokenizer:
- `claude`: `p = 0.000310`
- `hf-gpt2`: `p = 0.000039`
- `indicbert`: `p = 0.000225`
- `llama-3`: `p = 0.000039`
- `openai-gpt4o`: `p = 0.000138`
- `sarvam-105b`: `p = 0.004699`
- `sarvam-2b`: `p = 0.289718`
- `telugu-gpt2`: `p = 0.017051`
- `xlm-r`: `p = 0.067758`

So the between-category differences are strong for several tokenizers, but not uniformly strong across all of them.

---

## 6. Final status against the local revision plan

Implemented and rerun locally:
- dual-denominator fertility plumbing
- revised significance testing
- effect sizes
- minimal-pair variance outputs
- Kruskal-Wallis outputs
- `xlm-r`, `indicbert`, and `llama-3` tokenizer support
- direct English baseline collection and integration
- figure export including script fairness gap figure
- Sarvam log-likelihood MILU script

Still not fully closed:
- real Tenglish sourcing
- full manuscript-wide number refresh everywhere in the paper source
- completion of the long-running full MILU jobs now running in background logs
- final benchmark-correlation stage after those MILU outputs finish
