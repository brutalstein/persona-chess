# persona-chess

`persona-chess` is a Python library for training lightweight chess personas from PGN
files. The goal is not to find the strongest move. The goal is to predict how a
specific player is likely to move.

The current foundation includes PGN ingestion, player filtering, neural persona
training, legal move masking, checkpoint inference, baseline comparison models,
profile reports, JSON artifacts, and a CLI.

## Install

```bash
pip install persona-chess
```

That single command installs the runtime stack used by the main workflow:
PyTorch, Transformers, PEFT, compressed PGN support, and evaluation helpers.
When training starts, `persona-chess` checks the selected base model in the local
Hugging Face cache. If it is already present, it reuses it; if not, it downloads
it once and subsequent runs use the cached copy.

CUDA is handled through PyTorch. `persona-chess` uses CUDA automatically when the
installed PyTorch build can see a compatible GPU; otherwise it trains on CPU and
prints a diagnostic message. If CUDA is requested but unavailable, the error
explains whether the installed PyTorch wheel is CPU-only, whether CUDA is hidden,
or whether the requested CUDA device is invalid. For NVIDIA GPUs, use the
official PyTorch selector at https://pytorch.org/get-started/locally/ when you
need a specific CUDA wheel. A driver that reports CUDA 13.x support can still
need the CUDA wheel version currently published by PyTorch for Windows.
`persona-chess` will not silently rewrite the user's Python environment during
training; it reports the exact PyTorch/CUDA mismatch instead of guessing and
installing a different wheel behind the scenes.

For local development:

```bash
pip install -e ".[dev]"
```

## Python API

The normal workflow is train once, then load the checkpoint as a bot.

```python
from persona_chess import PersonaChess

result = PersonaChess().train(
    "games.pgn",
    player="Target Player",
    epochs=3,          # optional
    batch_size=32,     # optional
    device="cuda",     # optional; auto uses CUDA when PyTorch can see it
)

print(result.checkpoint_dir)
print(result.model_state_path)  # .../model.pt

bot = PersonaChess.load_neural(result.checkpoint_dir)
move = bot.move("startpos")
print(move.move_uci, move.san)
```

By default, persona checkpoints use `Maxlegrec/ChessBot` as the upstream base
policy. It is a MIT-licensed FEN-based Transformer chess model that predicts
moves from board positions. At inference time `bot.move(...)` blends that base
policy with the trained persona checkpoint, so the bot has a general chess prior
plus the selected player's PGN style. The base model is downloaded through
Hugging Face the first time it is needed after `pip install persona-chess`.

For large PGNs, keep the same API and switch on streaming. This writes training
records under the checkpoint folder and trains batch by batch instead of keeping
the whole dataset in memory:

```python
persona = PersonaChess()
result = persona.train(
    "large-games.pgn.zst",
    player="Target Player",
    streaming=True,
    validation_ratio=0.1,
)
```

Baseline personas are still available for quick comparison:

```python
from persona_chess import PersonaChess

persona = PersonaChess().fit_pgn("games.pgn", player="Target Player", model_type="blend")
persona.save("target-player.persona.json")
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

## Persona Evaluation

`persona-report` is the main product-level evaluation command. It keeps the
project-specific persona metrics in `persona-chess`, while optionally using common
scientific Python tooling through `persona-chess[evaluation]` when available.

The report includes move-match top-1/top-k, MRR, baseline deltas, phase metrics,
piece metrics, style similarity, opening similarity, score confidence, optional
SciPy distribution distances, and optional UCI-engine centipawn/blunder metrics.

```bash
persona-chess train games.pgn "Target Player" --model-type blend --out target.persona.json
persona-chess train games.pgn "Target Player" --model-type frequency --out baseline.persona.json
persona-chess persona-report target.persona.json games.pgn "Target Player" --baseline-model baseline.persona.json --out persona-report.json
persona-chess persona-report target.persona.json games.pgn "Target Player" --engine-path /path/to/stockfish --out engine-report.json
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
Input can be plain `.pgn` or compressed `.pgn.gz`, `.pgn.bz2`, `.pgn.xz`, and
`.pgn.zst` files. Zstandard support is installed by default.

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

There are two training modes:

- `PersonaChess().fit_pgn(...)` and `persona-chess train ...` fit a fast baseline
  persona artifact. This is the quickest way to create a playable opponent.
- `export-training-stream` plus `prepare-neural-stream` plus
  `train-neural-stream` runs the neural Transformer/LoRA path. Use this for a
  base model and stronger per-player adaptation.

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
