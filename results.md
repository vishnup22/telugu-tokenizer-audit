# Telugu Tokenizer Fertility & Fairness Audit — Complete Results

**Experiment:** `experiments/2026-06-23_v3_telugu_gpt2/`  
**Date:** 2026-06-23  
**Corpus:** 789 native_formal + 1,000 native_informal + 1,000 romanized_informal lines (2,789 total)  
**Tokenizers:** 6 — openai-gpt4o, claude (Haiku), hf-gpt2, sarvam-2b, sarvam-105b, telugu-gpt2  
**Minimal pairs:** 62 word pairs × 9 morphological types  
**Benchmark:** MILU Telugu test split, 7,304 questions

---

## 1. Fertility by Register (tokens/word)

Fertility = total tokens ÷ total whitespace-delimited words. Lower is more efficient.

| Tokenizer | Native Formal | Native Informal | Romanized Informal |
|---|---|---|---|
| openai-gpt4o | 3.28 | 3.25 | 3.98 |
| claude | 6.91 | **7.73** | 4.95 |
| hf-gpt2 | 19.99 | **22.08** | 4.74 |
| sarvam-2b | 2.52 | 3.01 | **5.67** |
| sarvam-105b | 2.68 | 2.71 | 4.01 |
| telugu-gpt2 | **1.68** | **1.60** | 5.49 |

### Fertility distribution (per-sentence, native informal)

| Tokenizer | Mean | Std | p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|---|---|---|
| openai-gpt4o | 3.29 | 0.64 | 2.83 | 3.21 | 3.68 | 4.11 | 5.09 |
| claude | 7.81 | 1.36 | 6.88 | 7.64 | 8.58 | 9.56 | 12.13 |
| hf-gpt2 | 22.28 | 3.55 | 19.86 | 21.83 | 24.60 | 26.89 | 31.60 |
| sarvam-2b | 3.06 | 1.03 | 2.33 | 2.83 | 3.54 | 4.40 | 6.17 |
| sarvam-105b | 2.76 | 0.66 | 2.30 | 2.67 | 3.14 | 3.65 | 4.70 |
| telugu-gpt2 | 1.63 | 0.42 | 1.33 | 1.55 | 1.80 | 2.13 | 3.00 |

### Token length distribution (% of words, native informal)

| Tokenizer | 1 tok | 2 tok | 3 tok | 4 tok | 5+ tok |
|---|---|---|---|---|---|
| openai-gpt4o | 4.3% | 10.0% | 28.7% | 27.2% | 29.8% |
| claude | 0.2% | 3.1% | 3.1% | 11.1% | **82.5%** |
| hf-gpt2 | 1.5% | 0.2% | 0.4% | 0.1% | **97.8%** |
| sarvam-2b | 12.5% | 24.4% | 28.4% | 19.3% | 15.4% |
| sarvam-105b | 8.1% | 27.2% | 33.6% | 19.0% | 12.1% |
| telugu-gpt2 | 26.3% | 43.4% | 21.3% | 6.7% | 2.3% |

**Key findings:**
- Claude assigns 5+ tokens to 82.5% of native Telugu words; GPT-2 assigns 5+ tokens to 97.8%
- Telugu-GPT2 keeps 69.7% of words at 1–2 tokens — most linguistically aligned
- Sarvam models distribute tokens evenly, reflecting deliberate Indic tokenizer design

---

## 2. Script Fairness Gap

Script fairness ratio = native_informal fertility ÷ romanized_informal fertility.  
Ratio < 1.0 → native script is cheaper. Ratio > 1.0 → native script is costlier (penalised).

| Tokenizer | Native Fertility | Romanized Fertility | Ratio | Verdict |
|---|---|---|---|---|
| openai-gpt4o | 3.25 | 3.98 | 0.82 | Native slightly cheaper |
| claude | 7.73 | 4.95 | **1.56** | **Native 56% costlier** |
| hf-gpt2 | 22.08 | 4.74 | **4.66** | **Native 4.7× costlier** |
| sarvam-2b | 3.01 | 5.67 | 0.53 | Native 47% cheaper |
| sarvam-105b | 2.71 | 4.01 | 0.68 | Native 32% cheaper |
| telugu-gpt2 | 1.60 | 5.49 | **0.29** | Native 71% cheaper |

### Wilcoxon signed-rank test (n=1,000 matched sentence pairs, native vs romanized)

| Tokenizer | Median gap | 95% CI | p-value | % pairs native costlier |
|---|---|---|---|---|
| openai-gpt4o | −0.73 | [−0.76, −0.69] | 1.0 (ns) | 7.7% |
| claude | +2.75 | [+2.71, +2.81] | **<10⁻¹⁶⁵** | **99.9%** |
| hf-gpt2 | +17.22 | [+17.0, +17.4] | **<10⁻¹⁶⁵** | **100.0%** |
| sarvam-2b | −2.65 | [−2.70, −2.58] | 1.0 (ns) | 0.0% |
| sarvam-105b | −1.30 | [−1.33, −1.25] | 1.0 (ns) | 0.2% |
| telugu-gpt2 | −3.88 | [−4.00, −3.83] | 1.0 (ns) | 0.0% |

**Key findings:**
- Claude and GPT-2 penalise native Telugu script at statistically significant levels (p<10⁻¹⁶⁵)
- GPT-2's 4.66× script gap is the most severe — native Telugu writers pay 4.7× more tokens than romanized
- Indic-trained tokenizers (Sarvam, Telugu-GPT2) do the opposite: native script is cheaper, rewarding users who write in their own script
- telugu-gpt2's 0.29 ratio means native script is 3.4× cheaper than romanized — the strongest native-script advantage

---

## 3. Minimal Pair Analysis

62 word pairs covering 9 Telugu morphological types, evaluated on native-script words only.

### Mean tokens per word by morphological type

| Morph Type | GPT-4o | Claude | GPT-2 | Sarvam-2B | Sarvam-105B | Telugu-GPT2 |
|---|---|---|---|---|---|---|
| base_noun | 2.6 | 4.4 | 12.4 | 2.1 | 2.0 | **1.0** |
| borrowed_base | 3.5 | 6.3 | 17.0 | 2.0 | 2.8 | **1.2** |
| borrowed_plus_case_suffix | 3.7 | 8.8 | 25.5 | 2.8 | 3.5 | **1.7** |
| case_suffix | 3.8 | 5.8 | 19.2 | 3.0 | 3.2 | **1.6** |
| case_suffix_with_sandhi | 3.7 | 7.3 | 23.4 | 2.5 | 3.0 | **1.7** |
| compound_with_sandhi | 4.8 | 9.0 | 27.0 | 3.3 | 4.2 | **1.8** |
| honorific_suffix | 5.7 | **11.7** | **35.5** | 2.8 | 4.5 | **2.2** |
| plural_suffix | 3.4 | 7.1 | 21.0 | 2.0 | 3.1 | **1.3** |
| verb_agglutination_chain | 4.6 | 8.8 | **29.6** | 2.0 | 3.5 | **1.8** |

**Key findings:**
- GPT-2 is catastrophically expensive on agglutinative forms: honorific_suffix averages 35.5 tokens/word, verb_agglutination_chain averages 29.6
- Claude struggles most with borrowed+case_suffix (8.8), honorifics (11.7), and verb chains (8.8) — morphologically complex forms are systematically over-segmented
- Telugu-GPT2 handles all forms in 1.0–2.2 tokens/word — within the expected morpheme-per-token range for a dedicated model
- Sarvam-2B is competitive with GPT-4o across all morphological types, often better

---

## 4. MILU Telugu Benchmark Accuracy

**Dataset:** ai4bharat/MILU, Telugu config, test split, 7,304 questions  
**Citation:** Verma et al. 2024 (arXiv:2411.02538)

| Model | Accuracy | Method | N questions | Source |
|---|---|---|---|---|
| openai-gpt4o | 72.53% | 5-shot MCQ | — | MILU paper Table 3 |
| claude-haiku-4-5 | **59.15%** | Zero-shot generative | 7,108/7,304 parseable | Our run |
| telugu-gpt2 (ours) | 26.97% | Log-likelihood (length-norm.) | 7,304/7,304 | Our run |
| Sarvam-2B | — | Zero-shot failed | 576/7,304 parseable | Our run (excluded) |

**Notes:**
- Sarvam-2B excluded: 92.1% of responses were unparseable — the model responded in Telugu rather than outputting A/B/C/D. The MILU paper reports 28.57% at 5-shot for reference only.
- telugu-gpt2 is the author's own model. Disclosed per standard practice. Near-chance accuracy (26.97% vs 25.0% baseline) is expected for a base LM with no instruction-following capability.

### Claude Haiku accuracy by domain

| Domain | Accuracy |
|---|---|
| Health & Medicine | 69.80% |
| Science | 66.74% |
| Engineering & Tech | 64.02% |
| Environmental Sciences | 61.79% |
| Business Studies | 60.04% |
| Social Sciences | 58.73% |
| Arts & Humanities | 54.95% |
| Law & Governance | 51.66% |

### Claude Haiku accuracy by subject (top 10 and bottom 5)

**Top 10:**
Information Technology 76.8%, Physics 74.6%, Religion & Spirituality 73.7%, Earth Sciences 73.2%, Chemistry 72.2%, Biology 70.1%, Food Science 70.0%, Health & Medicine 69.6%, Astronomy 68.4%, Finance 67.2%

**Bottom 5:**
Language Studies 46.6%, Business & Management 46.7%, Politics & Governance 48.2%, Education 48.5%, Transportation 49.5%

### telugu-gpt2 accuracy by subject (top 5 and bottom 5)

**Top 5:** Food Science 43.0%, Defense & Security 36.0%, Education 34.0%, Sociology 33.7%, Arts & Culture 33.2%

**Bottom 5:** Transportation 17.0%, Computer Science 17.5%, Agriculture 20.2%, Technology & Innovation 22.0%, Engineering 22.4%

---

## 5. Fertility–Accuracy Correlation

Data points used (instruction-tuned models only, consistent test set):

| Model | Native Informal Fertility | MILU Accuracy |
|---|---|---|
| openai-gpt4o | 3.25 tok/word | 72.53% (5-shot, paper) |
| claude-haiku-4-5 | 7.73 tok/word | 59.15% (zero-shot, our run) |
| telugu-gpt2 | 1.60 tok/word | 26.97% (log-likelihood, our run) |

**Finding:** No monotonic fertility→accuracy relationship. GPT-4o achieves highest accuracy with moderate fertility. Claude has the worst fertility but intermediate accuracy. Telugu-GPT2 has the best fertility but near-chance accuracy — because it is a base LM with no instruction following, not because its tokenizer is bad.

**Conclusion:** Tokenizer fertility and downstream capability are decoupled. The token tax is a **pricing equity problem**, not a capability signal. A model can have an excellent Telugu tokenizer (telugu-gpt2: 1.60) and still score at chance on a knowledge benchmark. Conversely, models with poor tokenizers (Claude: 7.73) can still understand Telugu well.

---

## 6. Cost & Context Window Implications

Based on native informal fertility (the real-world writing register for most users) and published pricing.

| Model | Fertility | Context limit | Effective Telugu words | Cost per 1,000 words |
|---|---|---|---|---|
| openai-gpt4o | 3.25 | 128,000 | ~39,385 | $0.0081 |
| claude-haiku | 7.73 | 200,000 | ~25,873 | $0.0062 |

**Token tax quantified:**
- A Telugu document of 10,000 words costs ~**$0.062** to process with Claude Haiku
- The same document in English (~10,000 tokens at ~1.0 tok/word) costs ~**$0.008**
- **Telugu speakers pay ~7.7× more** per word than English speakers using Claude Haiku
- Claude's 200K context fits only ~25,873 Telugu words vs ~200,000 English words — a **7.7× context shrinkage**

For GPT-4o, the token tax is ~3.25× (less severe but still significant).

---

## 7. Summary of All Key Numbers

| Metric | Value |
|---|---|
| **Highest fertility** | Claude, native informal: **7.73 tok/word** |
| **Lowest fertility** | Telugu-GPT2, native informal: **1.60 tok/word** |
| **Worst script gap** | GPT-2: **4.66×** native over romanized (p<10⁻¹⁶⁵, 100% of pairs) |
| **Best script gap (native favoured)** | Telugu-GPT2: **0.29×** (native 3.4× cheaper than romanized) |
| **Claude script gap** | **1.56×** — native Telugu 56% costlier, 99.9% of pairs (p<10⁻¹⁶⁵) |
| **Worst single morph type** | GPT-2, honorific_suffix: **35.5 tok/word** |
| **Best MILU accuracy** | GPT-4o: **72.53%** (5-shot, paper) |
| **Our best MILU run** | Claude Haiku: **59.15%** (zero-shot, 7,304 questions, 97.3% parseable) |
| **Base LM on MILU** | Telugu-GPT2: **26.97%** (log-likelihood, near chance=25%) |
| **Claude % words at 5+ tokens** | Native informal: **82.5%** |
| **GPT-2 % words at 5+ tokens** | Native informal: **97.8%** |
| **Telugu token tax (Claude)** | **~7.7× higher cost** per word vs English |
| **Effective context shrinkage** | Claude: 200K tokens → ~25,873 Telugu words (vs ~200K English words) |
| **Corpus size** | 2,789 lines, 3 registers |
| **Minimal pairs** | 62 word pairs, 9 morphological types |
| **Statistical test** | Wilcoxon signed-rank, n=1,000 pairs, bootstrap 95% CI |

---

## 8. Experimental Setup

| Component | Detail |
|---|---|
| Corpus registers | native_formal (formal Telugu prose), native_informal (conversational Telugu), romanized_informal (Telugu in Latin script / ITRANS) |
| Corpus size | 789 + 1,000 + 1,000 = 2,789 lines |
| Tokenizer APIs | Anthropic (claude-haiku-4-5), tiktoken (gpt-4o, gpt-2), HuggingFace transformers (sarvam-2b, sarvam-105b, telugu-gpt2 via SentencePiece) |
| Script gap test | Per-sentence Wilcoxon signed-rank on 1,000 matched native_informal vs romanized_informal pairs |
| Bootstrap CI | 1,000 resamples, 95% interval on median gap |
| MILU eval (Claude) | Zero-shot, assistant prefill (`"The answer is"`), max_tokens=50, greedy |
| MILU eval (telugu-gpt2) | Log-likelihood, length-normalised mean token log-prob, greedy argmax |
| MILU eval (Sarvam-2B) | Zero-shot generative — failed (92% unparseable), excluded |
| GPT-4o MILU score | From MILU paper Table 3, 5-shot evaluation |
