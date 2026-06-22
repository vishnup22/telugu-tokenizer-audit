# BUILD PLAN — Telugu Tokenizer Fertility & Fairness Audit

**Read this entire file before writing any code.** This is a build
specification for an AI coding agent (Cursor). It covers what to build,
in what order, with what interfaces — and, just as importantly, where to
stop and leave a clearly marked TODO for a human instead of inventing data.

The repo is laid out like a research lab's project, not a software
product: configs are separate from code, every experiment run is preserved
with its own config snapshot for provenance, exploratory notebooks are kept
apart from the pipeline that actually produces paper numbers, and the paper
source itself lives in the repo.

---

## 0. Ground rules for the agent

1. **Do not fabricate research data.** Never write placeholder Telugu
   sentences, benchmark accuracy numbers, or linguistic judgments and
   present them as if they were real. Every number that ends up in a
   results table must come from either (a) actually running code against
   real data/APIs, or (b) a cited published source. If a step needs data
   that doesn't exist yet, write the code to consume it, generate a clearly
   labeled `FAKE_FOR_TESTING` fixture to exercise the code path, and leave
   a `# TODO(human):` comment explaining exactly what real input is needed.
2. **Respect scraping legality.** Any data-collection code that touches a
   website must check robots.txt and ToS first and must strip personally
   identifying information (usernames, profile links) before storage —
   mirroring the privacy practice in the Bali et al. (2014) code-mixing
   paper this project is partly inspired by.
3. **Pin and log versions.** Tokenizers change behavior across library/model
   versions. Every run must log: library name + version, model/encoding
   name, and (for HF models) the specific revision/commit hash used. Write
   this to a `run_metadata.json` alongside every results file.
4. **Build in the milestone order in Section 6.** Don't jump ahead to
   reporting before the metrics layer has tests passing.
5. **Prefer boring, explicit code** over cleverness — this is a research
   tool that needs to be auditable by reviewers, not a production service.
6. **Write comments like a person, not a template.** Keep them short and
   plain — the kind of note someone jots down while actually working
   through the problem, not a restatement of the code.
   - No decorative banners or separators. Don't write `# ====...====`,
     `# ----...----`, `# ****...****`, or boxed section headers. A plain
     `#` line is enough if a section needs a heading at all.
   - Don't restate the function name in the comment (`# This function
     loads tokenizers` above `def load_tokenizers`).
   - Comment the non-obvious "why," skip the obvious "what."
   - Most lines don't need a comment. Don't comment every line out of habit.
   - Avoid stock phrasing like "This ensures that...", "This is responsible
     for...", "Note that...". Write it the way you'd say it out loud to a
     teammate.

   Avoid:
   ```python
   # ============================
   # SECTION: Fertility Calculation
   # ============================
   # This function is responsible for calculating the fertility metric,
   # which ensures that token-to-word ratios are properly computed.
   def compute_fertility(count_fn, lines):
   ```

   Prefer:
   ```python
   def compute_fertility(count_fn, lines):
       # whitespace word count is a rough stand-in -- Telugu doesn't
       # space-delimit postpositions the way English delimits words
   ```

   This applies to every file in `telugu_audit/`, `scripts/`, and `tests/`,
   not just the examples above.
7. **Never silently overwrite a previous run's results.** Every pipeline
   run gets its own timestamped folder under `experiments/`, with a snapshot
   of the exact config that produced it. This is what lets you (or a
   reviewer) trace any number back to exactly how it was produced.

---

## 1. Project goal

Audit how efficiently major LLM tokenizers handle Telugu, and show that the
answer depends on more than raw "fertility is high" — specifically:

- **Where** fragmentation happens linguistically (case/postposition suffixes,
  sandhi junctures, compounding, plural/honorific suffixation, and
  English-loanword + Telugu-suffix combinations).
- **Script effects**: whether the *same content* tokenizes more cheaply when
  written in Roman transliteration vs. native Telugu script — a fairness
  finding distinct from "Telugu is low-resource."
- **Downstream consequences**: correlation between fertility and accuracy on
  existing Telugu benchmarks, plus a real-dollar cost and effective-context-
  window translation.

Output: a set of reproducible, fully-provenanced experiment runs plus a
paper draft, suitable for a workshop submission (MRL, Indic NLP, or
code-switching/low-resource workshops at ACL/EMNLP/NAACL/COLING).

---

## 2. Repository structure

```
telugu-tokenizer-audit/
├── README.md
├── PLAN.md                             # this file
├── LICENSE
├── CITATION.cff                        # makes the repo citable once it's tied to a paper
├── pyproject.toml                      # package metadata + tool config (black/ruff/isort/pytest)
├── environment.yml                     # conda env, mirrors requirements.txt
├── requirements.txt
├── Makefile                            # make setup / corpus / audit / figures / paper
├── .env.example                        # ANTHROPIC_API_KEY=, HF_TOKEN=, OPENAI_API_KEY=
├── .gitignore                          # experiments/*/results, *.env, __pycache__/, .ipynb_checkpoints
├── .pre-commit-config.yaml             # black, ruff, isort, trailing-whitespace
│
├── configs/
│   ├── default.yaml                    # paths, output locations, which tokenizer set to use
│   ├── toy.yaml                        # points at tests/fixtures/toy_corpus for fast dev runs
│   ├── tokenizer_sets/
│   │   ├── full.yaml                   # every tokenizer adapter the project supports
│   │   └── quick_test.yaml             # 1-2 tokenizers, for fast iteration while developing
│   ├── pricing.yaml                    # $ per 1k tokens per model, dated (prices change)
│   └── benchmark_scores.example.yaml  # template for human-supplied benchmark accuracy numbers (see §7.5)
│
├── data/
│   ├── raw/                            # untouched scraped/downloaded source material
│   │   ├── native_formal/
│   │   ├── native_informal/
│   │   └── romanized_informal/
│   ├── interim/                        # partially cleaned, not yet final -- ok to be messy here
│   ├── processed/                      # final, one-sentence-per-line .txt, ready for the pipeline
│   │   ├── native_formal.txt
│   │   ├── native_informal.txt
│   │   └── romanized_informal.txt
│   └── minimal_pairs/
│       └── minimal_pairs.tsv           # word / gloss / morph_type
│
├── telugu_audit/                       # the installable package -- `pip install -e .`
│   ├── __init__.py
│   ├── run_utils.py                    # new_experiment_dir(), snapshot_config(), write_run_metadata(), load_yaml_config(), resolve_tokenizer_include()
│   ├── tokenizers/
│   │   ├── __init__.py
│   │   ├── registry.py                 # load_tokenizers(include=None)
│   │   └── adapters/
│   │       ├── __init__.py
│   │       ├── openai_adapter.py
│   │       ├── anthropic_adapter.py
│   │       ├── hf_adapter.py
│   │       └── stub_adapter.py         # key-free adapters used by tests; always loaded first
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── build.py                    # build_corpus(raw_dir, out_dir) -> {register: line_count}
│   │   ├── loaders.py                  # load_corpus(), load_minimal_pairs()
│   │   ├── cleaning.py                 # dedup, length filters, PII stripping
│   │   └── collectors/                 # human-gated scaffolding, see Section 7
│   │       ├── __init__.py
│   │       ├── wikipedia_dump.py
│   │       ├── social_media_TODO.py
│   │       └── transliteration_pairing_TODO.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── fertility.py                # fertility, compression, word_count
│   │   ├── parity.py                   # cross-language parity ratio
│   │   └── cost_translation.py         # $-per-content, effective context window
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── fertility_audit.py          # run_fertility_stage() — writes fertility_by_register.csv + script_fairness_gap.csv
│   │   ├── minimal_pair_audit.py       # run_minimal_pair_stage() — writes minimal_pair_fertility.csv
│   │   ├── minimal_pair_stats.py       # summarize_minimal_pairs(), flag_high_variance_morph_types()
│   │   └── fertility_accuracy_corr.py  # Pearson/Spearman vs. benchmark scores
│   └── reporting/
│       ├── __init__.py
│       ├── export.py                   # run_export_stage() — orchestrates tables, figures, and minimal_pair_summary.csv
│       ├── tables.py                   # Markdown + LaTeX table export
│       └── plots.py                    # matplotlib PNG/SVG figures
│
├── scripts/                            # thin entry points, numbered in run order
│   ├── 00_run_full_pipeline.py         # orchestrates stages 1–4 in sequence; --skip-corpus skips stage 1
│   ├── 01_build_corpus.py              # data/raw -> data/processed
│   ├── 02_run_tokenizer_audit.py       # register fertility + script-fairness gap
│   ├── 03_run_minimal_pair_audit.py
│   ├── 04_correlate_with_benchmarks.py
│   └── 05_make_figures_and_tables.py
│
├── notebooks/                          # exploration only -- nothing here feeds the paper directly
│   ├── 00_corpus_sanity_checks.ipynb
│   └── 01_explore_fertility_results.ipynb
│
├── experiments/                        # one self-contained folder per run (gitignored contents)
│   └── <YYYY-MM-DD>_<run_tag>/
│       ├── config_snapshot.yaml        # exact config used for this run
│       ├── run_metadata.json           # library/model versions, git commit, timestamps
│       ├── results/
│       │   ├── fertility_by_register.csv
│       │   ├── script_fairness_gap.csv
│       │   ├── minimal_pair_fertility.csv
│       │   └── minimal_pair_summary.csv    # morph_type aggregation, written by script 05
│       ├── figures/                    # PNG + SVG outputs written by script 05
│       └── tables/                     # Markdown + LaTeX exports written by script 05
│
├── tests/
│   ├── fixtures/
│   │   ├── test_config.yaml            # config pointing at toy_corpus for the end-to-end test
│   │   └── toy_corpus/                 # tiny synthetic data, filenames prefixed FAKE_
│   │       └── _processed/             # pre-built processed form of the FAKE_ files (avoids running script 01 in CI)
│   ├── test_cost_translation.py
│   ├── test_fertility.py
│   ├── test_minimal_pair_stats.py
│   ├── test_parity.py
│   ├── test_pipeline_end_to_end.py
│   ├── test_registry.py
│   └── test_run_utils.py
│
├── paper/
│   ├── main.tex
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_related_work.tex
│   │   ├── 03_methodology.tex
│   │   ├── 04_results.tex
│   │   ├── 05_discussion.tex
│   │   └── 06_conclusion.tex
│   ├── figures/                        # copied in from the experiments/ run used for submission
│   ├── tables/
│   ├── references.bib
│   └── outline.md                      # plain-language outline, kept alongside the LaTeX
│
└── docs/
    └── methodology_notes.md            # longer-form reasoning that doesn't belong in the paper
```

**Why this shape, not a generic `src/` layout:**
- `configs/` separates *what experiment you're running* from *how the code
  works*, so changing the tokenizer set or corpus path never means editing
  code.
- `experiments/<date>_<tag>/` means every number in the paper traces back to
  one folder containing the exact config and code-version metadata that
  produced it — nothing is ever silently overwritten by the next run.
- `scripts/` vs. `telugu_audit/` separates orchestration (thin, numbered,
  disposable) from logic (importable, unit-testable, reused across scripts
  and notebooks).
- `notebooks/` is explicitly exploratory and never authoritative — if a
  number from a notebook needs to go in the paper, it gets promoted into a
  real script/module first.
- `paper/` lives in the repo because the code and the writing should stay
  in sync, the same way an advisor would expect a student's repo to look
  right before a submission deadline.

---

## 3. Environment & dependencies

Set up with either conda or plain pip:

```bash
conda env create -f environment.yml && conda activate telugu-audit
# or
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pip install -e .          # installs telugu_audit as an importable package
```

`requirements.txt` (mirrored in `environment.yml`):
```
tiktoken
anthropic
transformers
sentencepiece
pandas
scipy
matplotlib
pyyaml
pytest
python-dotenv
```

`.env.example`:
```
ANTHROPIC_API_KEY=
HF_TOKEN=
OPENAI_API_KEY=
```

`pyproject.toml` should define package metadata (`name = "telugu_audit"`),
list the dependencies above, and hold tool config sections for `black`,
`ruff`, and `pytest` so formatting/linting is consistent without a person
having to remember flags.

`Makefile` targets:
```
make setup     # create env, install package + pre-commit hooks
make test      # pytest
make corpus    # scripts/01_build_corpus.py — build data/processed/ from data/raw/
make audit     # delegates to scripts/00_run_full_pipeline.py --skip-corpus
               #   (assumes corpus already built; runs stages 2–4 into a new experiments/ folder)
make figures EXPERIMENT_DIR=experiments/YYYY-MM-DD_mytag
               # scripts/05_make_figures_and_tables.py against an existing experiment folder;
               #   EXPERIMENT_DIR is required — e.g. make figures EXPERIMENT_DIR=experiments/2026-06-22_native-v1
make paper     # compile paper/main.tex (requires latexmk or similar)
```

Some HuggingFace models (Llama, Gemma) are gated — the human running this
needs to accept license terms on the Hub and set `HF_TOKEN` before those
specific adapters will work. Code must fail gracefully (skip + log a
warning) for any tokenizer it can't load, never crash the whole run.

---

## 4. Data contracts

### `configs/default.yaml`
```yaml
corpus_dir: data/processed
minimal_pairs_path: data/minimal_pairs/minimal_pairs.tsv
tokenizer_set: full              # references configs/tokenizer_sets/<name>.yaml
pricing_path: configs/pricing.yaml
run_tag: dev                     # overridden per real run, e.g. "native-vs-romanized-v1"
```

### `configs/tokenizer_sets/{full,quick_test}.yaml`
```yaml
include:
  - openai-gpt4o
  - claude
  # - llama-3
  # - mistral
  # ...
```

### `data/processed/{register}.txt`
Plain text, UTF-8, one sentence/post per line, no blank lines, no usernames
or other PII. `{register}` ∈ `{native_formal, native_informal,
romanized_informal}`.

### `data/minimal_pairs/minimal_pairs.tsv`
Tab-separated, header row required:
```
word    gloss    morph_type
```
`morph_type` is a closed vocabulary the analysis code should validate
against, e.g.: `base_noun`, `case_suffix`, `case_suffix_with_sandhi`,
`plural_suffix`, `honorific_suffix`, `compound_with_sandhi`,
`borrowed_base`, `borrowed_plus_case_suffix`, `verb_agglutination_chain`.
Reject (with a clear error, not a silent skip) any row with an
out-of-vocabulary `morph_type` — that's almost always a typo.

### `experiments/<date>_<run_tag>/results/fertility_by_register.csv`
Columns: `tokenizer, register, n_lines_attempted, n_lines_tokenized, total_tokens,
total_words, total_bytes, fertility_tokens_per_word, compression_bytes_per_token`

`n_lines_attempted` is the total number of lines passed to the tokenizer;
`n_lines_tokenized` is the count that succeeded. They differ only when a
tokenizer raises at runtime (e.g. a rate-limit or network error). Use
`n_lines_tokenized` as the denominator for any per-line average.

### `experiments/<date>_<run_tag>/results/script_fairness_gap.csv`
Columns: `tokenizer, native_fertility, romanized_fertility,
script_fertility_ratio_native_over_romanized`

### `experiments/<date>_<run_tag>/results/minimal_pair_fertility.csv`
Columns: `word, gloss, morph_type, tokenizer, n_tokens`

### `experiments/<date>_<run_tag>/results/minimal_pair_summary.csv`
Columns: `morph_type, tokenizer, n_words, mean_tokens, median_tokens, std_tokens`
Aggregated from `minimal_pair_fertility.csv` by `scripts/05` via
`telugu_audit/analysis/minimal_pair_stats.py`. Present only when script 03
has already run against the same experiment folder.

### `experiments/<date>_<run_tag>/run_metadata.json`
```json
{
  "timestamp": "...",
  "config_used": "config_snapshot.yaml",
  "tokenizer_versions": {"openai-gpt4o": {"library": "tiktoken==X.Y.Z", "encoding": "o200k_base"}, "...": "..."},
  "corpus_line_counts": {"native_formal": 0, "native_informal": 0, "romanized_informal": 0},
  "git_commit": "..."
}
```

---

## 5. Module specs

### 5.1 `telugu_audit/run_utils.py`
```python
def new_experiment_dir(run_tag: str, base_dir: str = "experiments") -> Path:
    """Creates experiments/<today>_<run_tag>/ and a results/ subfolder inside it.
    Errors out instead of overwriting if the exact folder already exists --
    pick a different run_tag instead of clobbering a previous run."""

def snapshot_config(config: dict, experiment_dir: Path) -> None:
    """Writes the resolved config dict to config_snapshot.yaml in the experiment dir."""

def write_run_metadata(experiment_dir: Path,
                       tokenizer_versions: dict,
                       corpus_line_counts: dict[str, int]) -> None:
    """Writes run_metadata.json: timestamp, config reference, tokenizer version
    dicts, corpus line counts, and current git commit hash (or "unknown" if not
    in a git repo)."""

def load_yaml_config(path: str | Path) -> dict:
    """Thin yaml.safe_load wrapper; used by scripts and tests so they don't
    each reimplement the same two lines."""

def resolve_tokenizer_include(config: dict, config_path: Path) -> set[str]:
    """Reads config['tokenizer_set'], locates configs/tokenizer_sets/<name>.yaml
    relative to config_path, and returns the set of tokenizer names to include."""
```

### 5.2 `telugu_audit/tokenizers/registry.py`
```python
CountFn = Callable[[str], int]

def load_tokenizers(include: set[str] | None = None) -> dict[str, CountFn]:
    """
    Loads every tokenizer adapter it can. On failure to load a given
    tokenizer, logs a warning and continues -- never raises.
    Returns name -> function mapping text to a token count (int).
    """

def get_tokenizer_versions() -> dict[str, dict]:
    """Returns version metadata for every tokenizer successfully loaded in the
    most recent load_tokenizers() call. Populated as a side-effect of that call
    so scripts can pass it straight to write_run_metadata()."""
```
Delegates to `adapters/stub_adapter.py`, `adapters/openai_adapter.py`,
`adapters/anthropic_adapter.py`, `adapters/hf_adapter.py`, each exposing
`get_tokenizers() -> tuple[dict[str, CountFn], dict[str, dict]]` (tokenizer
map plus version metadata) that `registry.py` merges. `stub_adapter` is
always loaded first and provides key-free adapters used by tests.

`adapters/anthropic_adapter.py` must use the real
`client.messages.count_tokens(model=..., messages=[...])` endpoint, reading
`resp.input_tokens` — this is a live API call (free, rate-limited), not a
local tokenizer file, so batch/cache calls where reasonable to avoid hitting
rate limits during a full corpus run.

### 5.3 `telugu_audit/metrics/fertility.py`
```python
def word_count(text: str) -> int:
    """Whitespace-split proxy. Telugu doesn't reliably space-delimit
    morphological words -- postpositions attach without a space -- so this
    is an approximation. Document this limitation wherever fertility is reported."""

def byte_count(text: str) -> int: ...

def compute_fertility(count_fn: CountFn, lines: list[str]) -> dict:
    """Returns n_lines, total_tokens, total_words, total_bytes,
    fertility_tokens_per_word, compression_bytes_per_token, aggregated
    across `lines`. n_lines is included so callers can reconstruct per-line
    averages without re-counting."""
```

### 5.4 `telugu_audit/metrics/parity.py`
```python
def parity_ratio(telugu_fertility: float, english_fertility: float) -> float:
    """telugu_fertility / english_fertility on matched-meaning sentence pairs.
    This is what makes cross-tokenizer comparison fair, since tokenizers
    differ in overall vocabulary size."""

def compute_parity_from_lines(count_fn: CountFn,
                               telugu_lines: list[str],
                               english_lines: list[str]) -> float:
    """Convenience wrapper: runs compute_fertility() on both lists, then calls
    parity_ratio(). Lists must be the same length (parallel corpus assumption)."""
```
Needs a matched Telugu-English parallel set (Samanantar or similar) — see
Section 7 for what's human-gated here.

### 5.5 `telugu_audit/metrics/cost_translation.py`
```python
def cost_per_content(n_tokens: int, price_per_1k_tokens: float) -> float: ...

def effective_context_words(context_limit_tokens: int, fertility: float) -> float:
    """How many Telugu words fit in a fixed token budget, given a tokenizer's
    measured fertility for that register/script."""
```
Reads prices from `configs/pricing.yaml`, not hardcoded constants — prices
change, and code shouldn't need editing when they do.

### 5.6 `telugu_audit/analysis/minimal_pair_stats.py`
Group `minimal_pair_fertility.csv` by `(tokenizer, morph_type)`, report mean/
median tokens-per-word per group, and flag which `morph_type` categories
show the largest tokenizer-to-tokenizer variance (these are the most
diagnostic for the paper's "where does fragmentation happen" claim).

### 5.7 `telugu_audit/analysis/fertility_accuracy_corr.py`
```python
def correlate_fertility_accuracy(fertility_by_model: dict[str, float],
                                  accuracy_by_model: dict[str, float]) -> dict:
    """Pearson + Spearman correlation, matching the Token Tax (AfriMMLU) method.
    `accuracy_by_model` must come from either a cited published benchmark
    result or a benchmark you actually ran -- never invented."""
```

### 5.8 `telugu_audit/reporting/export.py` / `tables.py` / `plots.py`
`export.py` is the stage-4 entry point called by `00_run_full_pipeline.py` and `scripts/05`:

```python
def run_export_stage(experiment_dir: Path) -> None:
    """Calls export_tables() and export_all_figures(), then — if
    minimal_pair_fertility.csv is present — runs summarize_minimal_pairs()
    and writes minimal_pair_summary.csv. Safe to call when optional result
    files are absent; those outputs are skipped silently."""
```

`tables.py` exports every results CSV as a Markdown table (for the paper
draft) and a LaTeX table. `plots.py` produces matplotlib PNG/SVG figures
(fertility-by-register bar chart, script-gap chart, fertility-vs-accuracy
scatter). All outputs go into the calling experiment's own `tables/` and
`figures/` subdirectories.

### 5.9 `scripts/`
Each script is a thin entry point: load a config, call into the
`telugu_audit` package, write outputs to a new `experiments/<date>_<tag>/`
folder via `run_utils.new_experiment_dir()`. If a script is doing anything
beyond reading a config, calling package functions, and writing files, that
logic belongs in the package, not the script.

```
python scripts/02_run_tokenizer_audit.py --config configs/default.yaml --run_tag native-vs-romanized-v1
```

### 5.10 `tests/`
- Unit tests for `fertility.py`, `parity.py`, `registry.py` (registry test
  should mock/stub tokenizer adapters — don't require real API keys to pass).
- One end-to-end test using `tests/fixtures/toy_corpus/` (tiny, clearly
  labeled `FAKE_*` files — a handful of made-up lines, not real Telugu
  corpus data) that runs `scripts/01` through `03` against a temp
  `experiments/` dir and checks the expected CSVs exist with the right columns.
- CI should be able to run the full test suite with zero API keys set
  (everything that needs a real key gets skipped, not failed).

---

## 6. Milestones (build in this order)

- [ ] **M1** — Repo scaffold: full directory tree, `pyproject.toml`,
  `environment.yml`, `requirements.txt`, `Makefile` skeleton, `.env.example`,
  `.gitignore`, `.pre-commit-config.yaml`, empty `__init__.py`s
- [ ] **M2** — `configs/` populated: `default.yaml`, `tokenizer_sets/full.yaml`
  and `quick_test.yaml`, a placeholder `pricing.yaml`
- [ ] **M3** — `telugu_audit/run_utils.py` + `metrics/fertility.py`, with
  tests passing on the toy fixture (no API keys needed)
- [ ] **M4** — `telugu_audit/tokenizers/registry.py` + adapters; verify first
  with `configs/tokenizer_sets/quick_test.yaml`, then expand to `full.yaml`
- [ ] **M5** — `telugu_audit/corpus/loaders.py` + `cleaning.py`; load the toy
  fixture end-to-end
- [ ] **M6** — `scripts/01` through `03` run start-to-finish on the toy
  fixture, producing a real `experiments/<date>_test/` folder with a config
  snapshot, `run_metadata.json`, and all three CSVs
- [ ] **M7** — `telugu_audit/metrics/parity.py` + `cost_translation.py`
- [ ] **M8** — `telugu_audit/analysis/` (minimal-pair stats + fertility/accuracy
  correlation)
- [ ] **M9** — `telugu_audit/reporting/` (tables + plots) + `scripts/05`
- [ ] **M10** — `notebooks/00` and `01` for exploratory sanity checks (these
  stay exploratory — nothing here is the source of truth for paper numbers)
- [ ] **M11** — Swap the toy fixture for real `data/processed/*.txt` once a
  human has supplied it (Section 7 gate), and run the full pipeline into a
  fresh, real-data `experiments/` run
- [ ] **M12** — `paper/` populated: `main.tex`, `sections/*.tex` stubs,
  `references.bib` seeded with the related-work citations from Section 8,
  figures/tables copied in from the latest real `experiments/` run

Do not start M11 until a human has confirmed the real corpus files and
expanded minimal-pairs list are in place — that's the gate described next.

---

## 7. What the agent cannot do — human-required steps

These need a person, not code, and should block M11 specifically:

1. **Collecting native_informal.txt and romanized_informal.txt as a matched
   pair.** This requires either finding naturally-occurring romanized posts
   on the same threads, or a native speaker transliterating a subset
   consistently. Mismatched content here invalidates the script-fairness
   finding.
2. **Validating and expanding `minimal_pairs.tsv`.** Needs a native Telugu
   speaker (ideally with linguistics background) to check every entry and
   grow it to 200-300+ rows across all `morph_type` categories.
3. **Scraping compliance review.** A human should check robots.txt/ToS for
   each actual source before any scraper in `corpus/collectors/` runs
   against it, and confirm PII stripping is sufficient.
4. **Gated model access.** Accepting HuggingFace license terms for Llama/
   Gemma, and obtaining any API keys, is a manual, account-bound step.
5. **Sourcing benchmark accuracy numbers.** Pulling real numbers from MATA,
   IndicXNLI, or the sentiment/hate-speech/sarcasm corpora — citing the
   actual papers — or running a benchmark subset yourself. The agent must
   never fill `accuracy_by_model` with guessed numbers.
6. **Keeping `configs/pricing.yaml` current.** API prices change; a human
   should refresh this file before any cost-translation numbers are
   reported, and note the date prices were checked.

The agent should implement everything around these gates (code, tests,
fixtures) so that the moment a human drops in real files, M11 is a single
command.

---

## 8. Paper outline scaffold (`paper/`)

`paper/outline.md` (plain-language version, written first):
1. Introduction/motivation
2. Related work — Petrov et al. (2023) tokenization premiums; Ahia et al.
   (2023/2024) socio-economic disparities; MEGA's Malayalam/Tamil fertility
   findings; the "Token Tax" AfriMMLU paper; IndicSuperTokenizer
3. Methodology — corpus, tokenizer set, metrics, minimal-pair design
4. Results — fertility/compression tables, the native-vs-romanized gap,
   minimal-pair breakdown by morph_type, cost/context translation,
   fertility-vs-accuracy correlation
5. Discussion — fairness/accessibility implications; who actually bears the
   "token tax" (informal/social-media users vs. formal register; native
   script vs. romanized writers)
6. Limitations — whitespace word-counting as a morphological-word proxy;
   corpus size/representativeness; tokenizer access gaps
7. Conclusion — optionally propose and evaluate a mitigation (vocabulary
   augmentation, Telugu-aware pre-tokenization)

`paper/sections/*.tex` mirrors this same structure once writing starts in
LaTeX; `paper/outline.md` stays as the plain-language reference so the
argument can be sanity-checked without compiling anything.

Target venues: MRL, Indic NLP, or code-switching/low-resource workshops at
ACL/EMNLP/NAACL/COLING.

---

## 9. Definition of done

The project is "built" when:
- `pytest` passes fully with zero API keys configured (using fixtures only)
- `scripts/01` through `05` run start-to-finish on real corpus data supplied
  by a human, producing a complete `experiments/<date>_<tag>/` folder
- `paper/outline.md` and `paper/sections/*.tex` have every results table/
  figure embedded with captions, pulled from one specific named experiment run
- Every number anywhere in `experiments/` or `paper/` traces back to either
  a real pipeline run or a cited source — nothing fabricated