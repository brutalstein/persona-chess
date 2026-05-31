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
persona-chess export-training-stream games.pgn "Target Player" --out target-player.train.jsonl
persona-chess export-base-training-stream public-games.pgn --out base.train.jsonl
persona-chess split-training-stream target-player.train.jsonl --train-out target-player.fit.jsonl --validation-out target-player.valid.jsonl
persona-chess split games.pgn "Target Player" --train-out train.jsonl --test-out test.jsonl
persona-chess benchmark games.pgn "Target Player" --model-type blend --out benchmark.json
persona-chess prepare-neural games.pgn "Target Player" --manifest-out adapter.manifest.json --move-vocab-out moves.vocab.json --position-vocab-out positions.vocab.json
persona-chess prepare-neural-stream target-player.train.jsonl "Target Player" --manifest-out adapter.manifest.json --move-vocab-out moves.vocab.json --position-vocab-out positions.vocab.json
persona-chess recommend-neural-config --training-examples 100000 --device cuda
persona-chess validate-neural adapter.manifest.json moves.vocab.json positions.vocab.json
persona-chess train-neural games.pgn "Target Player" --checkpoint-dir checkpoints/player --use-lora
persona-chess train-neural-stream target-player.train.jsonl --manifest adapter.manifest.json --move-vocab moves.vocab.json --position-vocab positions.vocab.json --checkpoint-dir checkpoints/player --init-checkpoint checkpoints/base
persona-chess neural-move checkpoints/player --fen "startpos"
persona-chess engine-move target-player.persona.json --engine-path /path/to/stockfish --fen "startpos"
persona-chess persona-report target-player.persona.json games.pgn "Target Player" --baseline-model baseline.persona.json --out persona-report.json
persona-chess download-model persona-chess/base-small --registry models.json
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

## Large PGN Training

For large PGN collections, use the streaming commands. They keep the training
records on disk and only materialize the active batch during neural training.

```bash
persona-chess export-training-stream games.pgn "Target Player" --out target-player.train.jsonl
persona-chess split-training-stream target-player.train.jsonl --train-out target-player.fit.jsonl --validation-out target-player.valid.jsonl
persona-chess prepare-neural-stream target-player.train.jsonl "Target Player" --manifest-out adapter.manifest.json --move-vocab-out moves.vocab.json --position-vocab-out positions.vocab.json
persona-chess train-neural-stream target-player.fit.jsonl --manifest adapter.manifest.json --move-vocab moves.vocab.json --position-vocab positions.vocab.json --checkpoint-dir checkpoints/player --validation-records target-player.valid.jsonl --use-lora
```

To build a general chess policy foundation before persona adaptation, export every
move from a large public PGN instead of filtering by one player:

```bash
persona-chess export-base-training-stream public-games.pgn --out base.train.jsonl
persona-chess split-training-stream base.train.jsonl --train-out base.fit.jsonl --validation-out base.valid.jsonl
persona-chess prepare-neural-stream base.fit.jsonl "persona-chess-base" --manifest-out base.manifest.json --move-vocab-out base.moves.json --position-vocab-out base.positions.json --config-profile large
persona-chess train-neural-stream base.fit.jsonl --manifest base.manifest.json --move-vocab base.moves.json --position-vocab base.positions.json --checkpoint-dir checkpoints/base --validation-records base.valid.jsonl --full-finetune
persona-chess train-neural-stream target-player.fit.jsonl --manifest adapter.manifest.json --move-vocab base.moves.json --position-vocab base.positions.json --checkpoint-dir checkpoints/player --validation-records target-player.valid.jsonl --init-checkpoint checkpoints/base --use-lora
```

Neural preparation uses a stable chess-wide position vocabulary by default so base
checkpoints and persona adapters can share the same embedding shapes. Use
`--data-position-vocab` only for isolated experiments that will not initialize from
another checkpoint.

## Neural Auto Configuration

Neural commands use hardware-aware defaults when training settings are omitted.
`persona-chess` inspects CPU memory and optional CUDA memory, then chooses a
small, balanced, or large Transformer + LoRA profile. Users can still override
training knobs directly:

```bash
persona-chess recommend-neural-config --training-examples 500000 --device cuda
persona-chess prepare-neural-stream target-player.train.jsonl "Target Player" --manifest-out adapter.manifest.json --move-vocab-out moves.vocab.json --position-vocab-out positions.vocab.json --config-profile balanced --epochs 3 --batch-size 32 --gradient-accumulation-steps 4
persona-chess train-neural-stream target-player.train.jsonl --manifest adapter.manifest.json --move-vocab moves.vocab.json --position-vocab positions.vocab.json --checkpoint-dir checkpoints/player --validation-records target-player.valid.jsonl --epochs 2 --batch-size 16
```

For very large datasets, keep the streaming path: export once, prepare the neural
artifacts from JSONL, then train from the manifest. This avoids loading the full
PGN or training set into memory at once.

The neural trainer reports train loss, optional validation loss, legal top-1
accuracy, legal top-3 accuracy, optimizer steps, active mixed precision mode, and
trainable parameter counts. Training uses AdamW, learning-rate warmup plus cosine
decay, gradient accumulation, gradient clipping, and CUDA mixed precision when
available.

Long neural runs can save resumable checkpoints:

```bash
persona-chess train-neural-stream target-player.fit.jsonl --manifest adapter.manifest.json --move-vocab moves.vocab.json --position-vocab positions.vocab.json --checkpoint-dir checkpoints/player --validation-records target-player.valid.jsonl --save-best --checkpoint-every-epoch
persona-chess train-neural-stream target-player.fit.jsonl --manifest adapter.manifest.json --move-vocab moves.vocab.json --position-vocab positions.vocab.json --checkpoint-dir checkpoints/player-resumed --resume-checkpoint checkpoints/player/best
```

Remote base checkpoints can be distributed as zip archives through a registry JSON.
Each archive must contain a `checkpoint.json` at or below its root.

```json
{
  "schema_version": "persona-chess/model-registry/v1",
  "models": [
    {
      "name": "persona-chess/base-small",
      "version": "0.1.0",
      "url": "https://example.com/persona-chess-base-small.zip",
      "sha256": "..."
    }
  ]
}
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
