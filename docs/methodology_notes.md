# Methodology notes

Longer-form reasoning that does not belong in the paper draft.

## Whitespace word counting

Telugu postpositions often attach without a space, so `word_count()` is a rough
proxy for morphological word count. Any fertility number derived from it should
say so explicitly in the paper.

## Script-fairness pairing

The native-vs-romanized comparison only holds if `native_informal.txt` and
`romanized_informal.txt` contain matched content. Mismatched pairs invalidate
the script gap metric entirely.

## Minimal pairs

`morph_type` labels must be validated by a native Telugu speaker before the
minimal-pair breakdown is publishable. The current `data/minimal_pairs/`
file is scaffolding only.

## Benchmark correlation

`accuracy_by_model` values must come from cited papers or runs you actually
performed. Script `04_correlate_with_benchmarks.py` expects a human-authored
YAML file.
