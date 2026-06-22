# Paper outline (plain language)

1. **Introduction** — Telugu speakers pay a tokenizer premium; fragmentation is
   not uniform across morphology and script.
2. **Related work** — Petrov et al. tokenization premiums; Ahia socio-economic
   disparities; MEGA Malayalam/Tamil fertility; Token Tax (AfriMMLU);
   IndicSuperTokenizer.
3. **Methodology** — three-register corpus, tokenizer set, fertility/compression
   metrics, minimal-pair design, cost/context translation.
4. **Results** — fertility tables, native-vs-romanized gap, morph_type breakdown,
   cost/context, fertility-vs-accuracy correlation.
5. **Discussion** — who bears the token tax (register, script choice).
6. **Limitations** — whitespace word proxy, corpus size, gated model access.
7. **Conclusion** — optional mitigation (vocab augmentation, Telugu-aware
   pre-tokenization).

Target venues: MRL, Indic NLP, code-switching/low-resource workshops at
ACL/EMNLP/NAACL/COLING.

## TODO(human)

- Replace placeholder results in `sections/04_results.tex` after M11 real-data run.
- Copy figures from a named `experiments/<date>_<run_tag>/` folder into
  `paper/figures/`.
