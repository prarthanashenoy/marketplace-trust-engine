.PHONY: install generate-data train evaluate serve dashboard test lint typecheck clean

install:
	poetry install

generate-data:
	poetry run python scripts/generate_data.py

train:
	poetry run python scripts/train.py

evaluate:
	poetry run python scripts/evaluate.py

serve:
	poetry run uvicorn src.marketplace_trust.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	poetry run streamlit run dashboard/app.py --server.port 8501

test:
	poetry run pytest tests/ -v --cov=src/marketplace_trust --cov-report=term-missing

lint:
	poetry run ruff check src/ tests/ scripts/
	poetry run ruff format --check src/ tests/ scripts/

typecheck:
	poetry run mypy src/marketplace_trust/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .ruff_cache/ .mypy_cache/
