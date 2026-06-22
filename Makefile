.PHONY: setup test corpus audit figures paper

PYTHON ?= python

setup:
	$(PYTHON) -m venv .venv
	.venv/Scripts/pip install -r requirements.txt
	.venv/Scripts/pip install -e ".[dev]"
	.venv/Scripts/pre-commit install

test:
	pytest

corpus:
	$(PYTHON) scripts/01_build_corpus.py --config configs/default.yaml

audit:
	$(PYTHON) scripts/00_run_full_pipeline.py --config configs/default.yaml --skip-corpus

figures:
	$(PYTHON) scripts/05_make_figures_and_tables.py --experiment-dir $(EXPERIMENT_DIR)

paper:
	latexmk -pdf paper/main.tex
