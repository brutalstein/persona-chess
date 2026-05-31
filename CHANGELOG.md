# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added engine-guided persona reranking with external UCI engines.

## 0.1.0 - 2026-05-31

- Added PGN ingestion, player filtering, persona profiling, and move datasets.
- Added baseline persona models: `blend`, `frequency`, `opening_book`, and `phase`.
- Added versioned persona, benchmark, neural manifest, vocabulary, and checkpoint artifacts.
- Added deterministic game-level train/test splits and benchmark reports.
- Added neural preparation pipeline with position tokenization, move vocabularies, policy batches, legal-masked training targets, optional PyTorch policy skeleton, and PEFT LoRA integration.
- Added neural checkpoint inference with legal move masking.
- Added CLI commands for profiling, dataset export, training, evaluation, benchmarking, neural preparation, neural validation, optional neural training, and neural checkpoint move prediction.
