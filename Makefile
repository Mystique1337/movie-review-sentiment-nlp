.PHONY: help setup data explore ablation classical curve transformer crossdomain errors demo all classical-only test lint clean

PYTHON ?= python3

help:
	@echo "Movie review sentiment analysis - reproduction targets"
	@echo ""
	@echo "  make setup          install dependencies"
	@echo "  make data           download the three corpora (~95 MB)"
	@echo "  make all            run the full experiment ladder (steps 1-8)"
	@echo "  make classical-only run everything except the transformer arm"
	@echo "  make test           run the unit tests"
	@echo ""
	@echo "Individual steps:"
	@echo "  make explore ablation classical curve transformer crossdomain errors demo"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/00_download_data.py

explore:
	$(PYTHON) scripts/01_explore_data.py

ablation:
	$(PYTHON) scripts/02_preprocessing_ablation.py

classical:
	$(PYTHON) scripts/03_train_classical.py

curve:
	$(PYTHON) scripts/04_learning_curve.py

transformer:
	$(PYTHON) scripts/05_train_transformer.py

crossdomain:
	$(PYTHON) scripts/06_cross_domain.py

errors:
	$(PYTHON) scripts/07_error_analysis.py

demo:
	$(PYTHON) scripts/08_demo_film_verdict.py

all: data explore ablation classical curve transformer crossdomain errors demo

classical-only: data explore ablation classical curve crossdomain errors demo

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	@echo "Cached corpora in data/ and trained models in models/ were left in place."
