# Trainable IDS Project

This is a multi-file defensive IDS prototype that supports:

- training on labeled datasets
- runtime prediction on logs
- optional LLM explanations
- SQLite alert storage
- FastAPI alert endpoint

## Project structure

```text
llm_ids_project/
├── app/
├── data/
├── models/
├── requirements.txt
├── train.py
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

## Run detector

```bash
python -m app.main
```

## Run API

```bash
uvicorn app.api:app --reload
```
