PYTHON ?= python
NPM ?= npm

.PHONY: dev test lint format benchmark docker-up docker-down install

install:
	$(PYTHON) -m pip install -e ".[birefnet,api,benchmark,test,dev]"
	cd frontend && $(NPM) ci

dev:
	$(PYTHON) -m uvicorn bg_removal.api:app --host 0.0.0.0 --port 8000 --reload & cd frontend && $(NPM) run dev

test:
	$(PYTHON) -m pytest
	cd frontend && $(NPM) test

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts
	cd frontend && $(NPM) run lint

format:
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m ruff format src tests scripts

benchmark:
	$(PYTHON) scripts/benchmark.py --device auto --repeats 3

docker-up:
	docker compose up --build

docker-down:
	docker compose down
