.PHONY: install test cov lint format pipeline gen1 gen2 gen3 gen4 gen5 db docs demo docker-build docker-test

install:
	pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
	pip install -r requirements.txt
	pip install -e . --no-deps
	pip install pytest-cov ruff pre-commit

test:
	python -m pytest -q

cov:
	python -m pytest --cov=src --cov-report=term-missing --cov-report=xml

lint:
	ruff check src tests experiments

format:
	ruff check src tests experiments --fix

pipeline:
	python -m experiments.run_all

gen1 gen2 gen3 gen4 gen5:
	python -m experiments.run_all --generation $(subst gen,,$@)

db:
	python -m experiments.stage29_research_db

docs:
	python -m src docs build

demo:
	python -m src demo

docker-build:
	docker build -t multi-asset-research .

docker-test: docker-build
	docker run --rm multi-asset-research
