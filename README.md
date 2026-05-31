# persona-chess

`persona-chess` is a Python library for training lightweight chess personas from PGN
files. The goal is not to find the strongest move. The goal is to predict how a
specific player is likely to move.

The current foundation includes PGN ingestion, player filtering, versioned move
datasets, game-level train/test splits, profile reports, baseline persona models,
JSON artifacts, and a CLI. The model layer is intentionally modular so a
chess-native Transformer + LoRA backend can be added without changing the product
surface.

## Install

```bash
pip install persona-chess
pip install "persona-chess[ml]"
```

For local development:

```bash
pip install -e ".[dev]"
```

## Python API

```python
from persona_chess import PersonaChess

persona = PersonaChess().fit_pgn("games.pgn", player="Target Player")
persona.save("target-player.persona.json")

prediction = persona.predict("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(prediction[0].san)
```

## CLI

```bash
persona-chess profile games.pgn "Target Player"
persona-chess model-card games.pgn "Target Player" --out target-player.model-card.json
persona-chess train games.pgn "Target Player" --model-type blend --out target-player.persona.json
persona-chess move target-player.persona.json --fen "startpos"
persona-chess export-training games.pgn "Target Player" --out target-player.train.jsonl
persona-chess split games.pgn "Target Player" --train-out train.jsonl --test-out test.jsonl
persona-chess benchmark games.pgn "Target Player" --model-type blend --out benchmark.json
persona-chess prepare-neural games.pgn "Target Player" --manifest-out adapter.manifest.json --move-vocab-out moves.vocab.json --position-vocab-out positions.vocab.json
persona-chess validate-neural adapter.manifest.json moves.vocab.json positions.vocab.json
persona-chess train-neural games.pgn "Target Player" --checkpoint-dir checkpoints/player --use-lora
persona-chess neural-move checkpoints/player --fen "startpos"
persona-chess engine-move target-player.persona.json --engine-path /path/to/stockfish --fen "startpos"
```

Built-in model backends:

- `blend`: weighted baseline combining exact position memory, opening book, and phase priors.
- `frequency`: exact position memory with global legal fallback.
- `opening_book`: early-game repertoire memory.
- `phase`: game-phase move prior for opening, middlegame, and endgame positions.

## Persona Model Cards

Model cards turn a PGN collection into a portable style and data-quality report.
They include style tags, move and phase breakdowns, data warnings, and a recommended
inference path.

```bash
persona-chess model-card games.pgn "Target Player" --out target-player.model-card.json
persona-chess model-card games.pgn "Target Player" --format markdown --out target-player.model-card.md
```

## Engine-Guided Persona Moves

`persona-chess` can rerank persona candidates with an external UCI engine such as
Stockfish or Lc0. The engine is not bundled and the persona model still supplies
the candidate style; the UCI engine only acts as a quality signal.

```bash
persona-chess engine-move target-player.persona.json --engine-path /path/to/stockfish --fen "startpos" --engine-weight 0.35
```

## Project Direction

The planned model path is:

1. A clean PGN-to-position dataset pipeline.
2. Deterministic train/test splitting at game level.
3. Strong baseline models for honest comparison.
4. A chess-native base Transformer trained on large public PGN data.
5. A training JSONL schema with legal move masks and target move indexes.
6. Per-player LoRA adapters trained from personal PGNs.
7. Legal move masking through `python-chess`.
8. Evaluation by move-match accuracy and style similarity metrics.

The neural command currently prepares versioned adapter manifests, move
vocabularies, and position vocabularies. The package also includes an optional
PyTorch Transformer policy skeleton behind the `ml` extra, with PEFT-powered LoRA,
legal-masked policy batches, checkpoint helpers, and checkpoint inference. It is not
enabled for standard installs.

## License

MIT
