#!/usr/bin/env python3
"""Generate Hindi minimal-pair candidates from UD Hindi-HDTB.

Output:
  data/minimal_pairs/hindi_minimal_pairs_candidates.tsv

This is a candidate-generation step only. The output is not a validated
experimental asset until a Hindi speaker reviews the words, glosses, and
category assignments and then curates a final Hindi-specific file.
"""

from __future__ import annotations

import csv
import urllib.request
from collections import defaultdict
from pathlib import Path


UD_HINDI_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Hindi-HDTB/master/hi_hdtb-ud-train.conllu"
)

VALID_MORPH_TYPES = [
    "base_noun",
    "oblique_case_form",
    "plural_form",
    "honorific_form",
    "borrowed_form",
    "verb_inflection_chain",
]

TARGET_PER_TYPE = 20


def _parse_feats(feats_str: str) -> dict[str, str]:
    if feats_str in ("_", ""):
        return {}
    return dict(part.split("=", 1) for part in feats_str.split("|") if "=" in part)


def _iter_tokens(conllu_text: str):
    for line in conllu_text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 10:
            continue
        tok_id = parts[0]
        if "-" in tok_id or "." in tok_id:
            continue
        yield {
            "form": parts[1],
            "lemma": parts[2],
            "upos": parts[3],
            "feats": _parse_feats(parts[5]),
        }


def _is_borrowed(form: str, feats: dict[str, str]) -> bool:
    if feats.get("Foreign") == "Yes":
        return True
    ascii_letters = sum(1 for ch in form if ord(ch) < 128 and ch.isalpha())
    total_letters = sum(1 for ch in form if ch.isalpha())
    return total_letters > 0 and ascii_letters / total_letters > 0.6


def _classify(tok: dict) -> str | None:
    form = tok["form"]
    upos = tok["upos"]
    feats = tok["feats"]

    if len(form) < 2 or upos in {"PUNCT", "SYM", "NUM", "X"}:
        return None

    if _is_borrowed(form, feats):
        return "borrowed_form"

    if upos in {"VERB", "AUX"} and feats.get("VerbForm") == "Fin":
        if feats.get("Tense") or feats.get("Aspect") or feats.get("Person"):
            return "verb_inflection_chain"

    if feats.get("Polite") == "Form" or form.endswith("जी"):
        return "honorific_form"

    if upos in {"NOUN", "PROPN"}:
        if feats.get("Number") == "Plur":
            return "plural_form"
        if feats.get("Case") and feats.get("Case") != "Nom":
            return "oblique_case_form"
        return "base_noun"

    return None


def _download_treebank() -> str:
    with urllib.request.urlopen(UD_HINDI_URL, timeout=30) as response:
        return response.read().decode("utf-8")


def main() -> None:
    out_path = Path("data/minimal_pairs/hindi_minimal_pairs_candidates.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = _download_treebank()
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for tok in _iter_tokens(text):
        morph_type = _classify(tok)
        if morph_type not in VALID_MORPH_TYPES:
            continue
        key = (tok["form"], morph_type)
        if key in seen:
            continue
        seen.add(key)
        by_type[morph_type].append(
            {
                "word": tok["form"],
                "gloss": tok["lemma"],
                "morph_type": morph_type,
                "source": "UD_Hindi-HDTB",
                "note": "lemma-as-gloss placeholder; requires Hindi-speaker review",
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# REVIEW BEFORE USE\n")
        handle.write("# Hindi-specific candidate set from UD Hindi-HDTB.\n")
        handle.write("# Categories are provisional and should not be mixed with Telugu categories without review.\n")
        writer = csv.DictWriter(
            handle,
            fieldnames=["word", "gloss", "morph_type", "source", "note"],
            delimiter="\t",
        )
        writer.writeheader()
        for morph_type in VALID_MORPH_TYPES:
            handle.write(f"# --- {morph_type} ---\n")
            for row in by_type[morph_type][:TARGET_PER_TYPE]:
                writer.writerow(row)

    total = sum(min(len(by_type[morph_type]), TARGET_PER_TYPE) for morph_type in VALID_MORPH_TYPES)
    print(f"Wrote {total} candidate rows to {out_path}")
    for morph_type in VALID_MORPH_TYPES:
        print(f"{morph_type}: {min(len(by_type[morph_type]), TARGET_PER_TYPE)} rows")


if __name__ == "__main__":
    main()
