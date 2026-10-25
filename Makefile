.PHONY: install test lint train evaluate app

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

train:
	python scripts/train.py --data data/cardio_train.xlsx

evaluate:
	python scripts/evaluate.py --data data/cardio_train.xlsx

app:
	streamlit run app.py
