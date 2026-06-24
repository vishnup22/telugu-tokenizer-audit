.PHONY: setup test corpus audit figures

PYTHON      ?= python
CONFIG      ?= configs/default.yaml
EXPERIMENT_DIR ?= experiments/run1
N_SAMPLES   ?= 1000

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

corpus:
	$(PYTHON) scripts/collect_data.py --n-samples $(N_SAMPLES)
	$(PYTHON) scripts/01_build_corpus.py --config $(CONFIG)

audit:
	$(PYTHON) scripts/02_run_tokenizer_audit.py --config $(CONFIG) --experiment-dir $(EXPERIMENT_DIR)
	$(PYTHON) scripts/03_run_minimal_pair_audit.py --config $(CONFIG) --experiment-dir $(EXPERIMENT_DIR)
	$(PYTHON) scripts/06_run_significance_tests.py --config $(CONFIG) --experiment-dir $(EXPERIMENT_DIR)
	$(PYTHON) scripts/05_make_figures_and_tables.py --experiment-dir $(EXPERIMENT_DIR)

figures:
	$(PYTHON) scripts/05_make_figures_and_tables.py --experiment-dir $(EXPERIMENT_DIR)
