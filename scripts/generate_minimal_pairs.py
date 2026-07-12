#!/usr/bin/env python3
"""Generate Telugu minimal pair candidates from UD treebank + Claude API.

Stage 1 — UD Telugu-MTG treebank (GitHub):
  Downloads te_mtg-ud-train.conllu and extracts morphologically annotated words,
  mapped to the 9 morph_type categories used by this project.

Stage 2 — Claude gap-fill:
  For any morph_type with fewer than TARGET_PER_TYPE candidates after Stage 1,
  prompts Claude to generate additional examples.

Output:
  data/minimal_pairs/minimal_pairs_candidates.tsv  -- draft for native-speaker review

Workflow after running this script:
  1. Open minimal_pairs_candidates.tsv
  2. Delete any incorrect rows (wrong gloss, wrong morph_type, non-Telugu text)
  3. Add any missing examples you know
  4. Save as data/minimal_pairs/minimal_pairs.tsv
  5. Run: python scripts/03_run_minimal_pair_audit.py --config configs/default.yaml \\
           --experiment-dir experiments/<your-dir>
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telugu_audit.run_utils import load_yaml_config                        

                                                                                 

UD_TELUGU_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Telugu-MTG/master/te_mtg-ud-train.conllu"
)

TARGET_PER_TYPE = 8                                                               

VALID_MORPH_TYPES = [
    "base_noun",
    "case_suffix",
    "case_suffix_with_sandhi",
    "plural_suffix",
    "honorific_suffix",
    "compound_with_sandhi",
    "borrowed_base",
    "borrowed_plus_case_suffix",
    "verb_agglutination_chain",
]

                                                                                 

def _parse_feats(feats_str: str) -> dict[str, str]:
    if feats_str in ("_", ""):
        return {}
    return dict(f.split("=") for f in feats_str.split("|") if "=" in f)


def _iter_tokens(conllu_text: str) -> Iterator[dict]:
    """Yield one dict per token (skip range/empty nodes)."""
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
            "form":  parts[1],
            "lemma": parts[2],
            "upos":  parts[3],
            "feats": _parse_feats(parts[5]),
            "misc":  parts[9],
        }


def _is_borrowed(tok: dict) -> bool:
    """Heuristic: foreign tag OR mostly ASCII characters in the form."""
    if tok["feats"].get("Foreign") == "Yes":
        return True
    form = tok["form"]
    ascii_ratio = sum(1 for c in form if ord(c) < 128) / max(len(form), 1)
    return ascii_ratio > 0.6


def _classify(tok: dict) -> str | None:
    """Map a UD token to one of the 9 morph_type values, or None to skip."""
    form  = tok["form"]
    upos  = tok["upos"]
    feats = tok["feats"]

                                              
    if upos in ("PUNCT", "NUM", "SYM") or len(form) < 2:
        return None

    case    = feats.get("Case", "")
    number  = feats.get("Number", "")
    polite  = feats.get("Polite", "")
    foreign = _is_borrowed(tok)

                                                                 
    if upos == "VERB" and feats.get("Tense") and feats.get("Person"):
        return "verb_agglutination_chain"

               
    if polite == "Form":
        return "honorific_suffix"

                    
    if foreign:
        if case:
            return "borrowed_plus_case_suffix"
        return "borrowed_base"

    if upos in ("NOUN", "PROPN"):
                
        if number == "Plur":
            return "plural_suffix"

        if case and case not in ("Nom", ""):
                                                                                    
            lemma_root = tok["lemma"].rstrip("ుాిీుూెేైొోౌం్")
            form_root  = form[:max(1, len(lemma_root))]
            sandhi = lemma_root and form_root and lemma_root != form_root
            return "case_suffix_with_sandhi" if sandhi else "case_suffix"

                          
        if not case or case == "Nom":
            return "base_noun"

                                                                               
    if upos in ("ADJ", "NOUN") and len(form) > 8 and "్" in form:
        return "compound_with_sandhi"

    return None


                                                                                

def extract_from_ud(url: str) -> list[dict]:
    print(f"Downloading UD Telugu treebank ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    print(f"  {text.count(chr(10))} lines downloaded")

    candidates: list[dict] = []
    seen: set[str] = set()

    for tok in _iter_tokens(text):
        morph_type = _classify(tok)
        if morph_type is None:
            continue
        key = (tok["form"], morph_type)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "word":       tok["form"],
            "gloss":      tok["lemma"],                                               
            "morph_type": morph_type,
            "source":     "UD_Telugu-MTG",
        })

    return candidates


                                                                                 

_CLAUDE_SYSTEM = """\
You are a Telugu linguistics expert. Generate Telugu word examples for a \
morphological minimal-pair dataset used in tokenizer research.

Rules:
- Use native Telugu script (Unicode).
- Each "word" entry is a single word or word form (no spaces unless a compound).
- "gloss" is a short English translation (2-5 words max).
- Only output a plain TSV block — no markdown, no headers, no explanations.
- Format: word<TAB>gloss<TAB>morph_type
"""

_TYPE_DESCRIPTIONS = {
    "base_noun":              "uninflected noun root (e.g. house, water, work)",
    "case_suffix":            "noun + case suffix, NO phonological sandhi at boundary",
    "case_suffix_with_sandhi":"noun + case suffix WITH phonological sandhi (stem changes)",
    "plural_suffix":          "noun + plural suffix",
    "honorific_suffix":       "noun or verb with honorific/polite suffix (-గారు, -వారు)",
    "compound_with_sandhi":   "two-morpheme compound with sandhi at the junction",
    "borrowed_base":          "English/Sanskrit/Persian loanword in isolation",
    "borrowed_plus_case_suffix": "loanword + Telugu case suffix (e.g. office-లో)",
    "verb_agglutination_chain":  "verb stem + tense + person/number agreement (finite verb form)",
}


def claude_generate(morph_type: str, n: int, existing: list[str]) -> list[dict]:
    """Ask Claude for n Telugu examples of the given morph_type."""
    import os
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    existing_str = ", ".join(existing[:5]) if existing else "none yet"

    prompt = (
        f"Generate {n} Telugu word examples for morph_type='{morph_type}'.\n"
        f"Description: {_TYPE_DESCRIPTIONS[morph_type]}\n"
        f"Already have these (do not repeat): {existing_str}\n\n"
        f"Output {n} lines, each: word<TAB>gloss<TAB>{morph_type}"
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=_CLAUDE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    rows = []
    for line in resp.content[0].text.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3:
            word, gloss, mt = parts
            if word and gloss:
                rows.append({"word": word, "gloss": gloss, "morph_type": mt, "source": "claude"})
    return rows


                                                                                 

def main() -> None:
    out_path = Path("data/minimal_pairs/minimal_pairs_candidates.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

                  
    candidates = extract_from_ud(UD_TELUGU_URL)
    by_type: dict[str, list[dict]] = {t: [] for t in VALID_MORPH_TYPES}
    for c in candidates:
        mt = c["morph_type"]
        if mt in by_type:
            by_type[mt].append(c)

    for mt, rows in by_type.items():
        print(f"  UD  {mt:35s} {len(rows):3d} candidates")

                               
    print("\nFilling gaps with Claude ...")
    for mt in VALID_MORPH_TYPES:
        current = by_type[mt]
        deficit = TARGET_PER_TYPE - len(current)
        if deficit <= 0:
            print(f"  OK  {mt:35s} (already {len(current)})")
            continue
        existing_words = [r["word"] for r in current]
        print(f"  GEN {mt:35s} (need {deficit} more) ...", end=" ", flush=True)
        new_rows = claude_generate(mt, deficit + 2, existing_words)             
        by_type[mt].extend(new_rows[:deficit])
        print(f"got {len(new_rows)}")

                  
    all_rows = [r for rows in by_type.values() for r in rows]
    header = "# REVIEW THIS FILE before renaming to minimal_pairs.tsv\n"
    header += "# Delete incorrect rows. Fix glosses. Add missing examples.\n"
    header += "# Column 'source': UD_Telugu-MTG = from treebank, claude = generated\n"
    header += "word\tgloss\tmorph_type\tsource\n"

    lines = [header]
    for mt in VALID_MORPH_TYPES:
        lines.append(f"# --- {mt} ---\n")
        for r in by_type[mt]:
            lines.append(f"{r['word']}\t{r['gloss']}\t{r['morph_type']}\t{r['source']}\n")

    out_path.write_text("".join(lines), encoding="utf-8")

    total = sum(len(v) for v in by_type.values())
    print(f"\nWrote {total} candidates to {out_path}")
    print("Next: review the file, then copy to data/minimal_pairs/minimal_pairs.tsv")
    print("      (remove the 'source' column before running the pipeline)")


if __name__ == "__main__":
    main()
