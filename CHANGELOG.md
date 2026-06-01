# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## 0.2.6 - 2026-06-01

- Added a mandatory default base-policy training preflight; training now verifies that `Maxlegrec/ChessBot` can load and return a legal move before starting.
- Added a `require_base_model` training flag that defaults to true for the Python API and neural training sessions.
- Exported `verify_hf_base_model_usable` for direct diagnostics.

## 0.2.5 - 2026-06-01

- Fixed Hugging Face base model loading with newer Transformers releases for remote models that do not define `all_tied_weights_keys`.
- Added an explicit `require_base_model` inference option so callers can fail loudly instead of falling back to persona-only neural inference.
- Added a stderr warning when base-model blending is requested but unavailable.

## 0.2.4 - 2026-06-01

- Added base-model cache preflight at training start; the default Hugging Face base model is downloaded once and reused from cache on later runs.
- Added a `prefetch_base_model` training option for controlled tests and advanced workflows.
- Documented that `persona-chess` reports CUDA wheel mismatches instead of silently rewriting the user's Python environment during training.

## 0.2.3 - 2026-06-01

- Added NVIDIA driver and `nvidia-smi` CUDA runtime detection when PyTorch is CPU-only.
- Added CUDA diagnostic tests for systems where the GPU is visible to Windows but not to the installed PyTorch wheel.
- Clarified CUDA install guidance for Windows NVIDIA systems.

## 0.2.2 - 2026-06-01

- Moved runtime ML, compressed PGN, and evaluation dependencies into the default install so `pip install persona-chess` is enough for the main workflow.
- Added stricter CUDA diagnostics for CPU-only PyTorch builds, hidden CUDA devices, invalid CUDA device requests, and unusable visible GPUs.
- Added clearer training and base-model loading messages with torch/CUDA runtime details.
- Raised the PyTorch requirement and added NVIDIA driver/CUDA runtime reporting to CUDA diagnostics.

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
