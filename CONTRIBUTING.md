# Contributing

Thanks for taking the project seriously. `persona-chess` aims to be a clean,
measurable, open-source Python package for chess persona modeling.

## Development Setup

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Install optional ML dependencies only when working on neural training:

```bash
.\.venv\Scripts\python -m pip install -e ".[ml]"
```

## Quality Bar

Before opening a pull request, run:

```bash
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m build
```

## Design Principles

- Keep chess rules outside neural prediction and enforce legal move masks.
- Prefer versioned artifacts over implicit file formats.
- Keep core installs lightweight; ML dependencies belong behind extras.
- Add benchmarks before claiming model improvements.
- Keep comments rare and useful.

