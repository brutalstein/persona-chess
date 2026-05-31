# Development

Create an isolated environment:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Run the quality suite:

```bash
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m build
.\.venv\Scripts\python -m twine check dist/*
```

Run a local smoke test:

```bash
.\.venv\Scripts\persona-chess benchmark tests\fixtures\sample.pgn "Target Player"
.\.venv\Scripts\persona-chess model-card tests\fixtures\sample.pgn "Target Player"
.\.venv\Scripts\persona-chess export-training tests\fixtures\sample.pgn "Target Player" --out sample.train.jsonl
.\.venv\Scripts\persona-chess prepare-neural tests\fixtures\sample.pgn "Target Player" --manifest-out adapter.manifest.json --move-vocab-out moves.vocab.json --position-vocab-out positions.vocab.json
.\.venv\Scripts\persona-chess validate-neural adapter.manifest.json moves.vocab.json positions.vocab.json
.\.venv\Scripts\persona-chess engine-move --help
```

Install optional ML dependencies:

```bash
.\.venv\Scripts\python -m pip install -e ".[ml]"
```

Run a neural training smoke test after installing ML dependencies:

```bash
.\.venv\Scripts\persona-chess train-neural tests\fixtures\sample.pgn "Target Player" --checkpoint-dir checkpoints\target-player --epochs 1 --batch-size 4 --use-lora
.\.venv\Scripts\persona-chess neural-move checkpoints\target-player --fen startpos
```

Install pre-commit hooks:

```bash
.\.venv\Scripts\pre-commit install
```

Build a local distribution:

```bash
.\.venv\Scripts\python -m build
```
