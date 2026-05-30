# Architecture

`persona-chess` keeps chess rules, data extraction, model inference, and artifact
storage separate. This makes the package useful today with a lightweight baseline
while leaving room for a Transformer + LoRA backend later.

## Layers

`pgn`
: Reads PGN files and filters games for a target player.

`dataset`
: Converts player games into supervised move examples and deterministic splits.

`training`
: Exports model-ready records with legal move masks and target move indexes.

`profile`
: Builds player-level summaries such as first-move and result distributions.

`models`
: Holds inference backends. The default `BlendPersonaModel` combines exact
position memory, opening-book memory, and phase-level priors. New backends should
register themselves through the model registry.

`evaluation`
: Measures move matching, coverage, rank quality, and prediction reasons.

`neural`
: Defines the future Transformer + LoRA surface: model configs, adapter manifests,
position tokenization, move vocabularies, position vocabularies, model-ready
policy samples, and an optional PyTorch backend.

`storage`
: Saves and loads versioned persona artifacts.

`facade`
: Provides the public Python API used by downstream applications and notebooks.

## Model Direction

The planned neural path is a chess-native policy model:

1. Train a base Transformer on large PGN-derived move records.
2. Keep legal move generation outside the model.
3. Fine-tune per-player LoRA adapters from personal PGNs.
4. Save small persona adapters as portable artifacts.
5. Evaluate with move-match and style-similarity metrics.

## Artifacts

The project uses versioned JSON artifacts for portability:

`persona`
: Saved user-facing model state.

`benchmark-report`
: Reproducible train/validation/test metrics for a model and PGN split.

`move-vocabulary`
: Global move ids for future neural policy heads.

`position-vocabulary`
: Token ids for chess-native position sequences.

`neural-adapter-manifest`
: Metadata for a future LoRA adapter, including base model, config, vocabulary sizes,
and training example count.

`neural-checkpoint`
: Metadata for optional PyTorch policy checkpoints and their companion artifacts.

## Neural Backend

The neural layer is split into pure-Python and optional PyTorch parts:

`tokens`
: Converts FEN positions into stable chess-native token sequences.

`position_vocabulary` and `vocabulary`
: Store input token ids and move ids as versioned JSON artifacts.

`samples`
: Converts training records into `input_ids`, `attention_mask`, legal move ids,
legal move masks, target move ids, and target legal indices.

`torch_backend`
: Builds a minimal Transformer policy model only when PyTorch is installed.

`lora`
: Applies LoRA through PEFT when optional ML dependencies are installed.

`trainer`
: Provides a small supervised training loop that computes loss over legal moves and
can train either LoRA adapters or the full policy skeleton.

`checkpoint`
: Saves and loads optional PyTorch policy checkpoints with adapter and vocabulary artifacts.

`validation`
: Checks neural artifact consistency before training or checkpointing.

## Baseline Strategy

Baseline models are part of the product, not throwaway code. They provide useful
behavior on small PGN files and create a stable comparison target for neural models.

`frequency`
: Memorizes exact positions and falls back to legal global move priors.

`opening_book`
: Focuses on early-game repertoire and transposition-compatible positions.

`phase`
: Learns legal move tendencies by side to move and game phase.

`blend`
: Combines the baseline family and is the default user-facing backend.
