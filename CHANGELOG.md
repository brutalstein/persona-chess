# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.2.1 - 2026-06-01

- Added real Hugging Face base-policy inference through `Maxlegrec/ChessBot`.
- Blended the downloaded FEN-based base chess policy with the trained persona checkpoint in `bot.move(...)`.
- Added `transformers` to the `ml` extra for base-model loading.
- Switched the default base model from a move-sequence-only model to a FEN-based chess model that matches the `move(fen)` API.

## 0.2.0 - 2026-06-01

- Added the primary neural bot workflow: `PersonaChess().train(...)`, `PersonaChess.load_neural(...)`, and `bot.move(...)`.
- Added automatic checkpoint directory creation with `model.pt`, checkpoint metadata, vocabularies, and training state.
- Added terminal training progress with epoch, batch, loss, elapsed time, ETA, device, and mixed precision.
- Added safer CUDA handling with automatic GPU use when supported and clear errors when CUDA is requested but unavailable.
- Selected `malcouffe/chessgpt` as the default upstream chess-language base model reference for future Hugging Face adapter training.
- Added hardware-aware neural auto configuration with user overrides for epochs, batch size, model size, and LoRA rank.
- Added validation metrics, optimizer-step tracking, gradient clipping, warmup/cosine scheduling, and CUDA mixed precision to neural training.
- Added all-player base-policy training export and deterministic streaming train/validation splits.
- Added stable neural position vocabularies and base-checkpoint initialization for persona fine-tuning.
- Added remote model registries, downloadable checkpoint archives, resumable neural training checkpoints, and best-epoch checkpointing.
- Added persona evaluation reports with baseline comparison, style similarity, opening similarity, and optional engine quality metrics.
- Added richer persona evaluation segments, confidence metrics, optional SciPy distribution distances, and compressed PGN readers.
- Added engine-guided persona reranking with external UCI engines.
- Added persona model cards with style summaries, data-quality warnings, and model recommendations.
- Added streaming PGN, JSONL, neural artifact, and policy-batch workflows for large training collections.

## 0.1.0 - 2026-05-31

- Added PGN ingestion, player filtering, persona profiling, and move datasets.
- Added baseline persona models: `blend`, `frequency`, `opening_book`, and `phase`.
- Added versioned persona, benchmark, neural manifest, vocabulary, and checkpoint artifacts.
- Added deterministic game-level train/test splits and benchmark reports.
- Added neural preparation pipeline with position tokenization, move vocabularies, policy batches, legal-masked training targets, optional PyTorch policy skeleton, and PEFT LoRA integration.
- Added neural checkpoint inference with legal move masking.
- Added CLI commands for profiling, dataset export, training, evaluation, benchmarking, neural preparation, neural validation, optional neural training, and neural checkpoint move prediction.
