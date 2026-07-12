# Telugu Tokenizer Fertility and Fairness Audit - Final Results Snapshot

**Primary latest run:** `experiments/2026-07-01_tenglish_r1/`  
**Main earlier run:** `experiments/2026-06-30_full_with_llama_english_r1/`  
**GPU benchmark run:** `experiments/2026-06-30_full_with_llama_english_r1_gpu/`  
**Corpus:** 789 `native_formal` + 1,000 `native_informal` + 1,000 `romanized_informal` + 1,000 `english_wiki` + 249 `tenglish_informal`  
**Primary word-count method:** `whitespace`  
**Sensitivity method also run:** `indicnlp`

---

## 1. Fertility by Register

| Tokenizer | Native Formal | Native Informal | Romanized Informal | English Wiki | Tenglish Informal |
|---|---:|---:|---:|---:|---:|
| `openai-gpt4o` | 3.28 | 3.25 | 3.98 | 1.33 | 1.65 |
| `claude` | 6.91 | 7.73 | 4.95 | 1.46 | 2.17 |
| `hf-gpt2` | 19.99 | 22.08 | 4.74 | 1.34 | 2.04 |
| `sarvam-2b` | 2.52 | 3.01 | 5.67 | 1.60 | 2.73 |
| `sarvam-105b` | 2.68 | 2.71 | 4.01 | 1.35 | 1.54 |
| `telugu-gpt2` | 1.68 | 1.60 | 5.49 | 2.65 | 2.59 |
| `xlm-r` | 2.50 | 2.41 | 3.93 | 1.52 | 1.62 |
| `indicbert` | 2.86 | 2.93 | 3.31 | 1.52 | 1.53 |
| `llama-3` | 12.96 | 14.15 | 4.30 | 1.35 | 1.88 |

Key takeaways:
- `telugu-gpt2` remains the most efficient tokenizer on native Telugu.
- `xlm-r`, `indicbert`, and both Sarvam tokenizers are much more efficient than Claude, Llama 3, and legacy GPT-2 on native Telugu.
- The new `tenglish_informal` sample is substantially cheaper than native Telugu for most tokenizers, but the gap is much smaller for `sarvam-2b` and reversed for `telugu-gpt2`.

---

## 2. Script Fairness Gap

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

Interpretation:
- English-heavy tokenizers can penalize native Telugu script strongly.
- Indic-trained or Telugu-aware tokenizers make native Telugu cheaper than the ITRANS baseline.

---

## 3. Telugu vs English

Measured multiplier from `native_informal / english_wiki`:

| Tokenizer | Multiplier |
|---|---:|
| `hf-gpt2` | 16.49x |
| `llama-3` | 10.49x |
| `claude` | 5.31x |
| `openai-gpt4o` | 2.44x |
| `sarvam-105b` | 2.01x |
| `indicbert` | 1.92x |
| `sarvam-2b` | 1.89x |
| `xlm-r` | 1.59x |
| `telugu-gpt2` | 0.60x |

This is the directly measured English baseline, not a borrowed estimate.

---

## 4. Tenglish vs Native

The organic Tenglish sample was built from 3,300 YouTube comments and filtered down to 249 lines.

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

Selected significance results from `tenglish_script_fairness_significance.json`:
- `openai-gpt4o`: median gap 1.76, CI [1.64, 1.82], rank-biserial 0.89
- `claude`: median gap 5.75, CI [5.57, 5.87], rank-biserial 0.97
- `hf-gpt2`: median gap 20.06, CI [19.78, 20.34], rank-biserial 1.00
- `sarvam-2b`: median gap 0.55, CI [0.33, 0.70], rank-biserial 0.24
- `telugu-gpt2`: median gap -0.93, CI [-1.02, -0.84], rank-biserial -0.84

---

## 5. Significance Outputs

The upgraded pipeline reports:
- two-sided Wilcoxon signed-rank tests
- median-gap direction
- rank-biserial correlation
- bootstrap 95% median-gap confidence intervals

Selected results from `script_gap_significance.json`:

| Tokenizer | Median gap | 95% CI | Rank-biserial | Direction |
|---|---:|---|---:|---|
| `openai-gpt4o` | -0.73 | [-0.76, -0.68] | -0.91 | romanized costlier |
| `claude` | 2.75 | [2.71, 2.81] | 1.00 | native costlier |
| `hf-gpt2` | 17.22 | [17.00, 17.42] | 1.00 | native costlier |
| `sarvam-2b` | -2.65 | [-2.70, -2.58] | -1.00 | romanized costlier |
| `sarvam-105b` | -1.30 | [-1.33, -1.25] | -1.00 | romanized costlier |
| `telugu-gpt2` | -3.88 | [-4.00, -3.83] | -1.00 | romanized costlier |
| `xlm-r` | -1.55 | [-1.60, -1.50] | -1.00 | romanized costlier |
| `indicbert` | -0.40 | [-0.40, -0.36] | -0.93 | romanized costlier |
| `llama-3` | 9.79 | [9.64, 9.92] | 1.00 | native costlier |

---

## 6. Morphological Summary

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

---

## 7. Correlation

The updated fertility-versus-accuracy correlation for the available benchmark set is written to:
- `experiments/2026-07-01_tenglish_r1/results/fertility_accuracy_correlation.json`

Current values:
- Pearson `r = 0.306`, `p = 0.802`
- Spearman `r = 0.500`, `p = 0.667`

The set is still small, so this remains descriptive rather than definitive.

---

## 8. MILU Results

Final MILU Telugu accuracy:

| Model | Method | Accuracy |
|---|---|---:|
| `openai-gpt4o` | paper-reported 5-shot | 72.53% |
| `claude` | local zero-shot generative | 60.92% |
| `sarvam-2b` | log-likelihood | 28.19% |
| `telugu-gpt2` | log-likelihood | 26.97% |
| chance baseline | - | 25.00% |

Additional notes:
- `claude` was resumed from checkpoint and finished at `7086 / 7304` parseable answers.
- `sarvam-2b` now has a full GPU log-likelihood run rather than a partial estimate.
- `telugu-gpt2` was the GPU Dravidian log-likelihood run.

---

## 9. Final Status

Implemented and rerun locally or on Colab:
- dual-denominator fertility plumbing
- revised significance testing
- effect sizes
- minimal-pair variance outputs
- Kruskal-Wallis outputs
- `xlm-r`, `indicbert`, and `llama-3` tokenizer support
- direct English baseline collection and integration
- Tenglish collection from public YouTube comments
- figure export including script fairness gap figure
- Sarvam and Dravidian MILU log-likelihood evaluation
- Claude MILU final completion at `60.92%`
- Tenglish rerun with source-backed organic comments

The Tenglish register is now in the repo artifacts and result files. Optional future work is just source expansion, not a blocking gap in the current revision.
