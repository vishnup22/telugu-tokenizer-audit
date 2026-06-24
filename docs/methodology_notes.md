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

In the current corpus, `romanized_informal.txt` is produced by ITRANS
transliteration of `native_informal.txt` via `indic-transliteration`. This
guarantees line alignment and morphological equivalence, but it is **not**
organic Tenglish (real users writing romanized Telugu ad hoc). ITRANS
romanization is systematic and reversible; organic romanized Telugu is
heterogeneous and code-switched. The paper should state this clearly in §3
and discuss it as a limitation in §6: fertility results for romanized may
differ when measured on real social-media romanized text.

## Claude token count overhead

The Anthropic `messages.count_tokens` API returns total input tokens including
a fixed message-format overhead (~11 tokens for a bare user message). Without
correction, Claude fertility numbers are inflated relative to tiktoken (which
counts raw text tokens with no overhead).

Fix applied in `anthropic_adapter.py`: the overhead is measured once at
startup with an empty message and subtracted from every subsequent count,
giving text-only token counts comparable across all adapters.

## Minimal pairs

`morph_type` labels must be validated by a native Telugu speaker before the
minimal-pair breakdown is publishable. The current `data/minimal_pairs/`
file is scaffolding only.

## Benchmark correlation

`accuracy_by_model` values must come from cited papers or runs you actually
performed. Script `04_correlate_with_benchmarks.py` expects a human-authored
YAML file.
