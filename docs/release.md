# Release

This project should ship small, honest releases. The first public package should
be treated as pre-alpha until neural checkpoints and benchmark reports are tested
on real PGN collections.

## Local Checks

```bash
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m build
.\.venv\Scripts\python -m twine check dist/*
```

## Neural Smoke

```bash
.\.venv\Scripts\persona-chess train-neural tests\fixtures\sample.pgn "Target Player" --checkpoint-dir checkpoints\target-player --epochs 1 --batch-size 4 --d-model 32 --n-layers 1 --n-heads 4 --use-lora --lora-rank 2 --lora-alpha 4
.\.venv\Scripts\persona-chess neural-move checkpoints\target-player --fen startpos
```

## PyPI

The `publish.yml` workflow is configured for PyPI Trusted Publishing. Before the
first release:

1. Confirm the package name is available.
2. Bump the version and move changelog entries out of `Unreleased`.
3. Create the project on PyPI and configure Trusted Publishing for the repository.
4. Create and push a signed version tag.
5. Create a GitHub release from that tag.
