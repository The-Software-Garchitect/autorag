.PHONY: install data test tune serve up report

PORT ?= 8091

install:
	python -m pip install -e ".[dev]"

data:
	python scripts/generate_dataset.py

test:
	python -m pytest

tune:
	python -m autorag tune --source data/acme

serve:
	python -m autorag serve --source data/acme --host 0.0.0.0 --port $(PORT)

up:
	docker compose up --build

report:
	python -m autorag eval --source data/acme
