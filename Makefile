.PHONY: install data data-scifact test tune serve up report index lint

PORT ?= 8091
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev,local,lite]"

data:
	$(PYTHON) scripts/generate_dataset.py

data-scifact:
	$(PYTHON) -m autorag fetch scifact

index:
	$(PYTHON) scripts/precompute_index.py

lint:
	ruff check autorag tests scripts

test:
	$(PYTHON) -m pytest

tune:
	$(PYTHON) -m autorag tune --source data/acme

serve:
	$(PYTHON) -m autorag serve --source data/acme --host 0.0.0.0 --port $(PORT)

up:
	docker compose up --build

report:
	python -m autorag eval --source data/acme
