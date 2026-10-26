PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard venv/bin/python),venv/bin/python,python3))

.PHONY: help install run test format lint clean data-check

help:
	@echo "TerraLens commands"
	@echo "  make install     Install the project and development tools"
	@echo "  make run         Open the dashboard on localhost:8501"
	@echo "  make test        Run the automated checks without downloading city data"
	@echo "  make data-check  Check whether real NYC source data is available"
	@echo "  make format      Format Python sources"
	@echo "  make lint        Check Python sources"
	@echo "  make clean       Remove project Python caches"

install:
	$(PYTHON) -m pip install -e '.[dev]'

run:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m isort app.py terralens/ scripts/ experiments/ --profile black
	$(PYTHON) -m black app.py terralens/ scripts/ experiments/ --line-length=100

lint:
	$(PYTHON) -m flake8 app.py terralens/ scripts/ experiments/ --max-line-length=100 --ignore=E203,W503

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__
	find terralens scripts experiments -type d -name "__pycache__" -prune -exec rm -rf {} +
	@echo "Project caches removed. Source data and predictions are preserved."

data-check:
	$(PYTHON) scripts/check_data_ready.py --city nyc
