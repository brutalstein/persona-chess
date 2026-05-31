import hashlib
import json
import random
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from persona_chess.dataset.builder import build_move_examples, iter_move_examples
from persona_chess.neural.autotune import NeuralAutoConfig, NeuralConfigProfile
from persona_chess.neural.autotune import recommend_neural_config as recommend_auto_config
from persona_chess.neural.checkpoint import (
    NeuralCheckpointManifest,
    load_torch_policy_state,
    load_torch_training_state,
    save_torch_policy_checkpoint,
)
from persona_chess.neural.config import (
    LoraConfig,
    MixedPrecisionMode,
    NeuralTrainingConfig,
    TransformerPolicyConfig,
)
from persona_chess.neural.manifest import AdapterManifest
from persona_chess.neural.model_hub import ModelRegistry, resolve_model_reference
from persona_chess.neural.planning import create_adapter_manifest_from_vocabulary_sizes
from persona_chess.neural.position_vocabulary import PositionVocabulary
from persona_chess.neural.samples import (
    PolicyBatch,
    PolicySample,
    build_policy_samples,
    collate_policy_samples,
    iter_policy_batches,
)
from persona_chess.neural.streaming import prepare_streaming_neural_artifacts
from persona_chess.neural.trainer import (
    TrainingEpochResult,
    TrainingProgressUpdate,
    TrainingResult,
    train_policy_model,
    train_policy_model_streaming,
)
from persona_chess.neural.vocabulary import MoveVocabulary
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.training import (
    TrainingRecord,
    build_training_records,
    iter_training_records,
    read_training_records_jsonl,
    write_training_records_jsonl,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class NeuralTrainRequest:
    pgn: str | Path
    player: str
    checkpoint_dir: str | Path
    base_model: str = "persona-chess/base-small"
    init_checkpoint: str | Path | None = None
    resume_checkpoint: str | Path | None = None
    model_registry: str | Path | None = None
    model_cache_dir: str | Path | None = None
    color: PlayerColor = "both"
    max_games: int | None = None
    skip_first_plies: int = 0
    use_lora: bool = True
    device: str | None = None
    config_profile: NeuralConfigProfile = "auto"
    standard_position_vocabulary: bool = True
    validation_ratio: float = 0.1
    streaming: bool = False
    records_dir: str | Path | None = None
    save_best: bool = True
    checkpoint_every_epoch: bool = False
    show_progress: bool = True
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    warmup_ratio: float | None = None
    max_grad_norm: float | None = None
    mixed_precision: MixedPrecisionMode | None = None
    gradient_accumulation_steps: int | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    dropout: float | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None


@dataclass(frozen=True, slots=True)
class NeuralRecordsTrainRequest:
    training_records: str | Path
    player: str
    checkpoint_dir: str | Path
    validation_records: str | Path | None = None
    base_model: str = "persona-chess/base-small"
    init_checkpoint: str | Path | None = None
    resume_checkpoint: str | Path | None = None
    model_registry: str | Path | None = None
    model_cache_dir: str | Path | None = None
    use_lora: bool = True
    device: str | None = None
    config_profile: NeuralConfigProfile = "auto"
    standard_position_vocabulary: bool = True
    save_best: bool = True
    checkpoint_every_epoch: bool = False
    show_progress: bool = True
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    warmup_ratio: float | None = None
    max_grad_norm: float | None = None
    mixed_precision: MixedPrecisionMode | None = None
    gradient_accumulation_steps: int | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    dropout: float | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None


@dataclass(frozen=True, slots=True)
class NeuralTrainResult:
    checkpoint_dir: Path
    checkpoint_manifest: NeuralCheckpointManifest
    adapter_manifest: AdapterManifest
    training_result: TrainingResult
    auto_config: NeuralAutoConfig
    training_examples: int
    validation_examples: int
    move_vocabulary: MoveVocabulary
    position_vocabulary: PositionVocabulary
    best_checkpoint_dir: Path | None = None
    epoch_checkpoint_dirs: tuple[Path, ...] = ()
    training_records: Path | None = None
    validation_records: Path | None = None

    @property
    def model_state_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint_manifest.model_state_file

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_state_path": str(self.model_state_path),
            "checkpoint_manifest": self.checkpoint_manifest.to_dict(),
            "adapter_manifest": self.adapter_manifest.to_dict(),
            "training_result": self.training_result.to_dict(),
            "auto_config": self.auto_config.to_dict(),
            "training_examples": self.training_examples,
            "validation_examples": self.validation_examples,
            "best_checkpoint_dir": (
                str(self.best_checkpoint_dir) if self.best_checkpoint_dir is not None else None
            ),
            "epoch_checkpoint_dirs": [str(path) for path in self.epoch_checkpoint_dirs],
            "training_records": (
                str(self.training_records) if self.training_records is not None else None
            ),
            "validation_records": (
                str(self.validation_records) if self.validation_records is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class _ResumeState:
    model_state_dict: dict[str, Any] | None = None
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    scaler_state_dict: dict[str, Any] | None = None
    next_epoch: int = 1


class _EpochCheckpointCallback:
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        adapter_manifest: AdapterManifest,
        move_vocabulary: MoveVocabulary,
        position_vocabulary: PositionVocabulary,
        lora_applied: bool,
        save_best: bool,
        checkpoint_every_epoch: bool,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.adapter_manifest = adapter_manifest
        self.move_vocabulary = move_vocabulary
        self.position_vocabulary = position_vocabulary
        self.lora_applied = lora_applied
        self.save_best = save_best
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.best_score: float | None = None
        self.best_checkpoint_dir: Path | None = None
        self.epoch_checkpoint_dirs: list[Path] = []
        self.latest_training_state: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        model: Any,
        epoch_result: TrainingEpochResult,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any,
    ) -> None:
        self.latest_training_state = _training_state_payload(
            epoch_result=epoch_result,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if self.checkpoint_every_epoch:
            saved = self._save_checkpoint(f"epoch-{epoch_result.epoch:04d}", model, epoch_result)
            self.epoch_checkpoint_dirs.append(saved)
        if self.save_best and self._is_best(epoch_result):
            self.best_checkpoint_dir = self._save_checkpoint("best", model, epoch_result)

    def _is_best(self, epoch_result: TrainingEpochResult) -> bool:
        score = (
            epoch_result.validation_loss
            if epoch_result.validation_loss is not None
            else epoch_result.average_train_loss
        )
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            return True
        return False

    def _save_checkpoint(
        self,
        name: str,
        model: Any,
        epoch_result: TrainingEpochResult,
    ) -> Path:
        directory = self.checkpoint_dir / name
        save_torch_policy_checkpoint(
            directory,
            model=model,
            adapter_manifest=self.adapter_manifest,
            move_vocabulary=self.move_vocabulary,
            position_vocabulary=self.position_vocabulary,
            training_result=_training_result_from_epoch(epoch_result),
            lora_applied=self.lora_applied,
            training_state=self.latest_training_state,
        )
        return directory


class _ConsoleTrainingProgress:
    def __init__(
        self,
        *,
        player: str,
        checkpoint_dir: Path,
        auto_config: NeuralAutoConfig,
        training_examples: int,
        validation_examples: int,
        use_lora: bool,
        stream: Any | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.last_rendered_at = 0.0
        self.finished = False
        hardware = auto_config.hardware
        device_detail = hardware.device_type
        if hardware.cuda_available and hardware.cuda_device_name:
            device_detail = (
                f"cuda ({hardware.cuda_device_name}, {hardware.cuda_memory_gb or 0:.1f} GB)"
            )
        print(
            "PersonaChess training: "
            f"player={player!r}, checkpoint={checkpoint_dir}, "
            f"device={device_detail}, epochs={auto_config.training.epochs}, "
            f"batch_size={auto_config.training.batch_size}, "
            f"grad_accum={auto_config.training.gradient_accumulation_steps}, "
            f"lora={'on' if use_lora else 'off'}, "
            f"train={training_examples}, validation={validation_examples}",
            file=self.stream,
            flush=True,
        )

    def __call__(self, update: TrainingProgressUpdate) -> None:
        now = time.monotonic()
        is_last = update.total_steps is not None and update.global_step >= update.total_steps
        if not is_last and now - self.last_rendered_at < 1.0:
            return

        self.last_rendered_at = now
        batch_text = (
            f"{update.batch}/{update.total_batches}"
            if update.total_batches is not None
            else str(update.batch)
        )
        step_text = (
            f"{update.global_step}/{update.total_steps}"
            if update.total_steps is not None
            else str(update.global_step)
        )
        eta_text = _format_duration(update.eta_seconds)
        line = (
            f"\rEpoch {update.epoch}/{update.total_epochs} | "
            f"batch {batch_text} | step {step_text} | "
            f"loss {update.loss:.4f} | "
            f"elapsed {_format_duration(update.elapsed_seconds)} | "
            f"eta {eta_text} | "
            f"device {update.device} | precision {update.mixed_precision}"
        )
        print(line, end="\n" if is_last else "", file=self.stream, flush=True)
        self.finished = is_last


def train_neural_persona(request: NeuralTrainRequest) -> NeuralTrainResult:
    if request.streaming:
        records_dir = (
            Path(request.records_dir)
            if request.records_dir
            else _default_records_dir(request.checkpoint_dir)
        )
        train_path, validation_path, _, validation_count = _write_pgn_training_records(
            request,
            records_dir=records_dir,
        )
        result = train_neural_records(
            NeuralRecordsTrainRequest(
                training_records=train_path,
                validation_records=validation_path if validation_count else None,
                player=request.player,
                checkpoint_dir=request.checkpoint_dir,
                base_model=request.base_model,
                init_checkpoint=request.init_checkpoint,
                resume_checkpoint=request.resume_checkpoint,
                model_registry=request.model_registry,
                model_cache_dir=request.model_cache_dir,
                use_lora=request.use_lora,
                device=request.device,
                config_profile=request.config_profile,
                standard_position_vocabulary=request.standard_position_vocabulary,
                save_best=request.save_best,
                checkpoint_every_epoch=request.checkpoint_every_epoch,
                show_progress=request.show_progress,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                warmup_ratio=request.warmup_ratio,
                max_grad_norm=request.max_grad_norm,
                mixed_precision=request.mixed_precision,
                gradient_accumulation_steps=request.gradient_accumulation_steps,
                d_model=request.d_model,
                n_layers=request.n_layers,
                n_heads=request.n_heads,
                dropout=request.dropout,
                lora_rank=request.lora_rank,
                lora_alpha=request.lora_alpha,
                lora_dropout=request.lora_dropout,
            )
        )
        return result

    examples = build_move_examples(
        request.pgn,
        GameFilter(player=request.player, color=request.color, max_games=request.max_games),
        skip_first_plies=request.skip_first_plies,
    )
    records = build_training_records(examples)
    train_records, validation_records = _split_validation_records(
        records,
        validation_ratio=request.validation_ratio,
        seed=42,
    )
    return _train_neural_records_in_memory(
        train_records,
        validation_records=validation_records,
        request=request,
    )


def train_neural_records(request: NeuralRecordsTrainRequest) -> NeuralTrainResult:
    train_path = Path(request.training_records)
    validation_path = Path(request.validation_records) if request.validation_records else None
    artifacts = prepare_streaming_neural_artifacts(
        read_training_records_jsonl(train_path),
        standard_position_vocabulary=request.standard_position_vocabulary,
    )
    validation_examples = (
        sum(1 for _ in read_training_records_jsonl(validation_path))
        if validation_path is not None
        else 0
    )
    if artifacts.training_examples <= 0:
        raise ValueError("training_records must contain at least one record")

    auto_config = _resolve_neural_auto_config(
        artifacts.training_examples,
        config_profile=request.config_profile,
        device=request.device,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        warmup_ratio=request.warmup_ratio,
        max_grad_norm=request.max_grad_norm,
        mixed_precision=request.mixed_precision,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        d_model=request.d_model,
        n_layers=request.n_layers,
        n_heads=request.n_heads,
        dropout=request.dropout,
        lora_rank=request.lora_rank,
        lora_alpha=request.lora_alpha,
        lora_dropout=request.lora_dropout,
    )
    manifest = create_adapter_manifest_from_vocabulary_sizes(
        player=request.player,
        base_model=request.base_model,
        transformer=auto_config.transformer,
        lora=auto_config.lora,
        training=auto_config.training,
        move_vocabulary_size=artifacts.move_vocabulary.size,
        position_vocabulary_size=artifacts.position_vocabulary.size,
        training_examples=artifacts.training_examples,
    )
    model, result, callback = _run_streaming_training(
        train_path,
        validation_path=validation_path,
        validation_examples=validation_examples,
        request=request,
        manifest=manifest,
        auto_config=auto_config,
        move_vocabulary=artifacts.move_vocabulary,
        position_vocabulary=artifacts.position_vocabulary,
    )
    checkpoint_dir = Path(request.checkpoint_dir)
    checkpoint = save_torch_policy_checkpoint(
        checkpoint_dir,
        model=model,
        adapter_manifest=manifest,
        move_vocabulary=artifacts.move_vocabulary,
        position_vocabulary=artifacts.position_vocabulary,
        training_result=result,
        lora_applied=request.use_lora,
        training_state=callback.latest_training_state,
    )
    return NeuralTrainResult(
        checkpoint_dir=checkpoint_dir,
        checkpoint_manifest=checkpoint,
        adapter_manifest=manifest,
        training_result=result,
        auto_config=auto_config,
        training_examples=artifacts.training_examples,
        validation_examples=validation_examples,
        move_vocabulary=artifacts.move_vocabulary,
        position_vocabulary=artifacts.position_vocabulary,
        best_checkpoint_dir=callback.best_checkpoint_dir,
        epoch_checkpoint_dirs=tuple(callback.epoch_checkpoint_dirs),
        training_records=train_path,
        validation_records=validation_path,
    )


def write_pgn_training_records(
    pgn: str | Path,
    out: str | Path,
    *,
    player: str,
    color: PlayerColor = "both",
    max_games: int | None = None,
    skip_first_plies: int = 0,
) -> int:
    examples = iter_move_examples(
        pgn,
        GameFilter(player=player, color=color, max_games=max_games),
        skip_first_plies=skip_first_plies,
    )
    return write_training_records_jsonl(out, iter_training_records(examples))


def _train_neural_records_in_memory(
    train_records: list[TrainingRecord],
    *,
    validation_records: list[TrainingRecord],
    request: NeuralTrainRequest,
) -> NeuralTrainResult:
    if not train_records:
        raise ValueError("PGN did not produce any training records for this player")

    auto_config = _resolve_neural_auto_config(
        len(train_records),
        config_profile=request.config_profile,
        device=request.device,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        warmup_ratio=request.warmup_ratio,
        max_grad_norm=request.max_grad_norm,
        mixed_precision=request.mixed_precision,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        d_model=request.d_model,
        n_layers=request.n_layers,
        n_heads=request.n_heads,
        dropout=request.dropout,
        lora_rank=request.lora_rank,
        lora_alpha=request.lora_alpha,
        lora_dropout=request.lora_dropout,
    )
    move_vocabulary = MoveVocabulary.standard()
    position_vocabulary = _position_vocabulary_for_records(
        train_records,
        standard_position_vocabulary=request.standard_position_vocabulary,
    )
    manifest = create_adapter_manifest_from_vocabulary_sizes(
        player=request.player,
        base_model=request.base_model,
        transformer=auto_config.transformer,
        lora=auto_config.lora,
        training=auto_config.training,
        move_vocabulary_size=move_vocabulary.size,
        position_vocabulary_size=position_vocabulary.size,
        training_examples=len(train_records),
    )
    initial_state_dict, resume_state = _load_training_states(
        init_checkpoint=request.init_checkpoint,
        resume_checkpoint=request.resume_checkpoint,
        model_registry=request.model_registry,
        model_cache_dir=request.model_cache_dir,
        transformer=auto_config.transformer,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        device=request.device,
        use_lora=request.use_lora,
    )
    batches = [
        collate_policy_samples(batch, move_pad_id=move_vocabulary.pad_id)
        for batch in _chunks(
            _samples_for_records(
                train_records,
                transformer=auto_config.transformer,
                move_vocabulary=move_vocabulary,
                position_vocabulary=position_vocabulary,
            ),
            auto_config.training.batch_size,
        )
    ]
    validation_batches = (
        [
            collate_policy_samples(batch, move_pad_id=move_vocabulary.pad_id)
            for batch in _chunks(
                _samples_for_records(
                    validation_records,
                    transformer=auto_config.transformer,
                    move_vocabulary=move_vocabulary,
                    position_vocabulary=position_vocabulary,
                ),
                auto_config.training.batch_size,
            )
        ]
        if validation_records
        else None
    )
    callback = _build_epoch_checkpoint_callback(
        checkpoint_dir=Path(request.checkpoint_dir),
        adapter_manifest=manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        lora_applied=request.use_lora,
        save_best=request.save_best,
        checkpoint_every_epoch=request.checkpoint_every_epoch,
    )
    progress_callback = _build_progress_callback(
        show_progress=request.show_progress,
        player=request.player,
        checkpoint_dir=Path(request.checkpoint_dir),
        auto_config=auto_config,
        training_examples=len(train_records),
        validation_examples=len(validation_records),
        use_lora=request.use_lora,
    )
    model, result = train_policy_model(
        batches,
        transformer=auto_config.transformer,
        training=auto_config.training,
        position_vocabulary_size=position_vocabulary.size,
        move_vocabulary_size=move_vocabulary.size,
        device=request.device,
        lora=auto_config.lora if request.use_lora else None,
        validation_batches=validation_batches,
        initial_state_dict=initial_state_dict,
        resume_state_dict=resume_state.model_state_dict,
        optimizer_state_dict=resume_state.optimizer_state_dict,
        scheduler_state_dict=resume_state.scheduler_state_dict,
        scaler_state_dict=resume_state.scaler_state_dict,
        start_epoch=resume_state.next_epoch,
        epoch_callback=callback,
        progress_callback=progress_callback,
    )
    checkpoint_dir = Path(request.checkpoint_dir)
    checkpoint = save_torch_policy_checkpoint(
        checkpoint_dir,
        model=model,
        adapter_manifest=manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        training_result=result,
        lora_applied=request.use_lora,
        training_state=callback.latest_training_state,
    )
    return NeuralTrainResult(
        checkpoint_dir=checkpoint_dir,
        checkpoint_manifest=checkpoint,
        adapter_manifest=manifest,
        training_result=result,
        auto_config=auto_config,
        training_examples=len(train_records),
        validation_examples=len(validation_records),
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        best_checkpoint_dir=callback.best_checkpoint_dir,
        epoch_checkpoint_dirs=tuple(callback.epoch_checkpoint_dirs),
    )


def _run_streaming_training(
    train_path: Path,
    *,
    validation_path: Path | None,
    validation_examples: int,
    request: NeuralRecordsTrainRequest,
    manifest: AdapterManifest,
    auto_config: NeuralAutoConfig,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
) -> tuple[Any, TrainingResult, _EpochCheckpointCallback]:
    initial_state_dict, resume_state = _load_training_states(
        init_checkpoint=request.init_checkpoint,
        resume_checkpoint=request.resume_checkpoint,
        model_registry=request.model_registry,
        model_cache_dir=request.model_cache_dir,
        transformer=auto_config.transformer,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        device=request.device,
        use_lora=request.use_lora,
    )
    callback = _build_epoch_checkpoint_callback(
        checkpoint_dir=Path(request.checkpoint_dir),
        adapter_manifest=manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        lora_applied=request.use_lora,
        save_best=request.save_best,
        checkpoint_every_epoch=request.checkpoint_every_epoch,
    )

    def batch_factory() -> Iterable[PolicyBatch]:
        return iter_policy_batches(
            read_training_records_jsonl(train_path),
            position_vocabulary=position_vocabulary,
            move_vocabulary=move_vocabulary,
            max_sequence_length=auto_config.transformer.max_sequence_length,
            batch_size=auto_config.training.batch_size,
        )

    validation_batch_factory: Callable[[], Iterable[PolicyBatch]] | None = None
    if validation_path is not None:

        def validation_batch_factory() -> Iterable[PolicyBatch]:
            return iter_policy_batches(
                read_training_records_jsonl(validation_path),
                position_vocabulary=position_vocabulary,
                move_vocabulary=move_vocabulary,
                max_sequence_length=auto_config.transformer.max_sequence_length,
                batch_size=auto_config.training.batch_size,
            )

    progress_callback = _build_progress_callback(
        show_progress=request.show_progress,
        player=request.player,
        checkpoint_dir=Path(request.checkpoint_dir),
        auto_config=auto_config,
        training_examples=manifest.training_examples,
        validation_examples=validation_examples,
        use_lora=request.use_lora,
    )
    model, result = train_policy_model_streaming(
        batch_factory,
        transformer=auto_config.transformer,
        training=auto_config.training,
        position_vocabulary_size=position_vocabulary.size,
        move_vocabulary_size=move_vocabulary.size,
        device=request.device,
        lora=auto_config.lora if request.use_lora else None,
        initial_state_dict=initial_state_dict,
        resume_state_dict=resume_state.model_state_dict,
        optimizer_state_dict=resume_state.optimizer_state_dict,
        scheduler_state_dict=resume_state.scheduler_state_dict,
        scaler_state_dict=resume_state.scaler_state_dict,
        start_epoch=resume_state.next_epoch,
        validation_batch_factory=validation_batch_factory,
        training_batches=_count_batches(
            manifest.training_examples, auto_config.training.batch_size
        ),
        epoch_callback=callback,
        progress_callback=progress_callback,
    )
    return model, result, callback


def _write_pgn_training_records(
    request: NeuralTrainRequest,
    *,
    records_dir: Path,
) -> tuple[Path, Path, int, int]:
    _require_validation_ratio(request.validation_ratio)
    records_dir.mkdir(parents=True, exist_ok=True)
    train_path = records_dir / "train.records.jsonl"
    validation_path = records_dir / "validation.records.jsonl"
    train_count = 0
    validation_count = 0
    examples = iter_move_examples(
        request.pgn,
        GameFilter(player=request.player, color=request.color, max_games=request.max_games),
        skip_first_plies=request.skip_first_plies,
    )
    with (
        train_path.open("w", encoding="utf-8") as train_handle,
        validation_path.open("w", encoding="utf-8") as validation_handle,
    ):
        for record in iter_training_records(examples):
            line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            score = _stream_split_score(record_key=_record_split_key(record), seed=42)
            if request.validation_ratio > 0 and score < request.validation_ratio:
                validation_handle.write(line)
                validation_count += 1
            else:
                train_handle.write(line)
                train_count += 1
    if train_count == 0:
        raise ValueError("PGN did not produce any training records for this player")
    return train_path, validation_path, train_count, validation_count


def _resolve_neural_auto_config(
    training_examples: int,
    *,
    config_profile: NeuralConfigProfile,
    device: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    warmup_ratio: float | None,
    max_grad_norm: float | None,
    mixed_precision: MixedPrecisionMode | None,
    gradient_accumulation_steps: int | None,
    d_model: int | None,
    n_layers: int | None,
    n_heads: int | None,
    dropout: float | None,
    lora_rank: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
) -> NeuralAutoConfig:
    auto_config = recommend_auto_config(
        training_examples,
        profile=config_profile,
        device=device,
    )
    transformer = TransformerPolicyConfig(
        max_sequence_length=auto_config.transformer.max_sequence_length,
        d_model=d_model if d_model is not None else auto_config.transformer.d_model,
        n_layers=n_layers if n_layers is not None else auto_config.transformer.n_layers,
        n_heads=n_heads if n_heads is not None else auto_config.transformer.n_heads,
        dropout=dropout if dropout is not None else auto_config.transformer.dropout,
    )
    training = NeuralTrainingConfig(
        epochs=epochs if epochs is not None else auto_config.training.epochs,
        batch_size=batch_size if batch_size is not None else auto_config.training.batch_size,
        learning_rate=(
            learning_rate if learning_rate is not None else auto_config.training.learning_rate
        ),
        weight_decay=auto_config.training.weight_decay,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else auto_config.training.gradient_accumulation_steps
        ),
        warmup_ratio=warmup_ratio
        if warmup_ratio is not None
        else auto_config.training.warmup_ratio,
        max_grad_norm=(
            max_grad_norm if max_grad_norm is not None else auto_config.training.max_grad_norm
        ),
        mixed_precision=(
            mixed_precision if mixed_precision is not None else auto_config.training.mixed_precision
        ),
        seed=auto_config.training.seed,
    )
    lora = LoraConfig(
        rank=lora_rank if lora_rank is not None else auto_config.lora.rank,
        alpha=lora_alpha if lora_alpha is not None else auto_config.lora.alpha,
        dropout=lora_dropout if lora_dropout is not None else auto_config.lora.dropout,
        target_modules=auto_config.lora.target_modules,
    )
    overrides = _neural_override_names(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    notes = (f"Manual overrides applied: {', '.join(overrides)}.",) if overrides else ()
    return auto_config.with_configs(
        transformer=transformer,
        training=training,
        lora=lora,
        notes=notes,
    )


def _load_training_states(
    *,
    init_checkpoint: str | Path | None,
    resume_checkpoint: str | Path | None,
    model_registry: str | Path | None,
    model_cache_dir: str | Path | None,
    transformer: TransformerPolicyConfig,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
    device: str | None,
    use_lora: bool,
) -> tuple[dict[str, Any] | None, _ResumeState]:
    if init_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("init_checkpoint and resume_checkpoint cannot be used together")
    registry = ModelRegistry.load(model_registry) if model_registry is not None else None
    initial_state_dict = _load_initial_policy_state(
        init_checkpoint,
        transformer=transformer,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        device=device,
        registry=registry,
        cache_dir=model_cache_dir,
    )
    resume_state = _load_resume_policy_state(
        resume_checkpoint,
        transformer=transformer,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        device=device,
        use_lora=use_lora,
        registry=registry,
        cache_dir=model_cache_dir,
    )
    return initial_state_dict, resume_state


def _load_initial_policy_state(
    init_checkpoint: str | Path | None,
    *,
    transformer: TransformerPolicyConfig,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
    device: str | None,
    registry: ModelRegistry | None,
    cache_dir: str | Path | None,
) -> dict[str, Any] | None:
    if init_checkpoint is None:
        return None

    checkpoint_path = resolve_model_reference(
        init_checkpoint, registry=registry, cache_dir=cache_dir
    )
    state_dict, checkpoint_manifest, adapter_manifest, checkpoint_moves, checkpoint_positions = (
        load_torch_policy_state(checkpoint_path, device=device)
    )
    if checkpoint_manifest.lora_applied:
        raise ValueError("init_checkpoint must be a full/base checkpoint saved without LoRA")
    _require_checkpoint_compatibility(
        adapter_manifest=adapter_manifest,
        transformer=transformer,
        checkpoint_moves=checkpoint_moves,
        move_vocabulary=move_vocabulary,
        checkpoint_positions=checkpoint_positions,
        position_vocabulary=position_vocabulary,
        label="init checkpoint",
    )
    return state_dict


def _load_resume_policy_state(
    resume_checkpoint: str | Path | None,
    *,
    transformer: TransformerPolicyConfig,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
    device: str | None,
    use_lora: bool,
    registry: ModelRegistry | None,
    cache_dir: str | Path | None,
) -> _ResumeState:
    if resume_checkpoint is None:
        return _ResumeState()

    checkpoint_path = resolve_model_reference(
        resume_checkpoint,
        registry=registry,
        cache_dir=cache_dir,
    )
    state_dict, checkpoint_manifest, adapter_manifest, checkpoint_moves, checkpoint_positions = (
        load_torch_policy_state(checkpoint_path, device=device)
    )
    if checkpoint_manifest.lora_applied != use_lora:
        raise ValueError("resume checkpoint LoRA mode does not match this run")
    _require_checkpoint_compatibility(
        adapter_manifest=adapter_manifest,
        transformer=transformer,
        checkpoint_moves=checkpoint_moves,
        move_vocabulary=move_vocabulary,
        checkpoint_positions=checkpoint_positions,
        position_vocabulary=position_vocabulary,
        label="resume checkpoint",
    )
    training_state = load_torch_training_state(checkpoint_path, device=device)
    next_epoch = int(training_state.get("epoch", 0)) + 1 if training_state else 1
    return _ResumeState(
        model_state_dict=state_dict,
        optimizer_state_dict=training_state.get("optimizer_state_dict"),
        scheduler_state_dict=training_state.get("scheduler_state_dict"),
        scaler_state_dict=training_state.get("scaler_state_dict"),
        next_epoch=max(1, next_epoch),
    )


def _require_checkpoint_compatibility(
    *,
    adapter_manifest: AdapterManifest,
    transformer: TransformerPolicyConfig,
    checkpoint_moves: MoveVocabulary,
    move_vocabulary: MoveVocabulary,
    checkpoint_positions: PositionVocabulary,
    position_vocabulary: PositionVocabulary,
    label: str,
) -> None:
    if adapter_manifest.transformer.to_dict() != transformer.to_dict():
        raise ValueError(f"{label} transformer config does not match this run")
    if checkpoint_moves.id_to_token != move_vocabulary.id_to_token:
        raise ValueError(f"{label} move vocabulary does not match this run")
    if checkpoint_positions.id_to_token != position_vocabulary.id_to_token:
        raise ValueError(f"{label} position vocabulary does not match this run")


def _build_epoch_checkpoint_callback(
    *,
    checkpoint_dir: Path,
    adapter_manifest: AdapterManifest,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
    lora_applied: bool,
    save_best: bool,
    checkpoint_every_epoch: bool,
) -> _EpochCheckpointCallback:
    return _EpochCheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        adapter_manifest=adapter_manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        lora_applied=lora_applied,
        save_best=save_best,
        checkpoint_every_epoch=checkpoint_every_epoch,
    )


def _build_progress_callback(
    *,
    show_progress: bool,
    player: str,
    checkpoint_dir: Path,
    auto_config: NeuralAutoConfig,
    training_examples: int,
    validation_examples: int,
    use_lora: bool,
) -> Callable[[TrainingProgressUpdate], None] | None:
    if not show_progress:
        return None
    return _ConsoleTrainingProgress(
        player=player,
        checkpoint_dir=checkpoint_dir,
        auto_config=auto_config,
        training_examples=training_examples,
        validation_examples=validation_examples,
        use_lora=use_lora,
    )


def _samples_for_records(
    records: Iterable[TrainingRecord],
    *,
    transformer: TransformerPolicyConfig,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
) -> list[PolicySample]:
    return build_policy_samples(
        records,
        position_vocabulary=position_vocabulary,
        move_vocabulary=move_vocabulary,
        max_sequence_length=transformer.max_sequence_length,
    )


def _training_state_payload(
    *,
    epoch_result: TrainingEpochResult,
    optimizer: Any,
    scheduler: Any | None,
    scaler: Any,
) -> dict[str, Any]:
    return {
        "epoch": epoch_result.epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
    }


def _training_result_from_epoch(epoch_result: TrainingEpochResult) -> TrainingResult:
    return TrainingResult(
        epochs=1,
        steps=epoch_result.steps,
        final_loss=epoch_result.average_train_loss,
        optimizer_steps=epoch_result.optimizer_steps,
        average_train_loss=epoch_result.average_train_loss,
        validation_loss=epoch_result.validation_loss,
        validation_accuracy=epoch_result.validation_accuracy,
        validation_top3_accuracy=epoch_result.validation_top3_accuracy,
        validation_examples=epoch_result.validation_examples,
        best_validation_loss=epoch_result.validation_loss,
        best_epoch=epoch_result.epoch,
        mixed_precision="off",
        epochs_detail=(epoch_result,),
    )


def _position_vocabulary_for_records(
    records: list[TrainingRecord],
    *,
    standard_position_vocabulary: bool,
) -> PositionVocabulary:
    if standard_position_vocabulary:
        return PositionVocabulary.standard()
    return PositionVocabulary.from_records(records)


def _split_validation_records(
    records: list[T],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[T], list[T]]:
    _require_validation_ratio(validation_ratio)
    if validation_ratio == 0 or len(records) < 2:
        return records, []

    validation_size = max(1, int(round(len(records) * validation_ratio)))
    validation_size = min(validation_size, len(records) - 1)
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_size])
    train_records = [
        record for index, record in enumerate(records) if index not in validation_indices
    ]
    validation_records = [
        record for index, record in enumerate(records) if index in validation_indices
    ]
    return train_records, validation_records


def _require_validation_ratio(validation_ratio: float) -> None:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")


def _record_split_key(record: TrainingRecord) -> str:
    return f"{record.game_index}:{record.ply}:{record.position_key}:{record.target_move}"


def _stream_split_score(*, record_key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{record_key}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") / float(2**64)


def _neural_override_names(
    *,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    warmup_ratio: float | None,
    max_grad_norm: float | None,
    mixed_precision: MixedPrecisionMode | None,
    gradient_accumulation_steps: int | None,
    d_model: int | None,
    n_layers: int | None,
    n_heads: int | None,
    dropout: float | None,
    lora_rank: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
) -> list[str]:
    values: list[tuple[str, object | None]] = [
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("learning_rate", learning_rate),
        ("warmup_ratio", warmup_ratio),
        ("max_grad_norm", max_grad_norm),
        ("mixed_precision", mixed_precision),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("d_model", d_model),
        ("n_layers", n_layers),
        ("n_heads", n_heads),
        ("dropout", dropout),
        ("lora_rank", lora_rank),
        ("lora_alpha", lora_alpha),
        ("lora_dropout", lora_dropout),
    ]
    return [name for name, value in values if value is not None]


def _count_batches(examples: int, batch_size: int) -> int:
    return max(1, (examples + batch_size - 1) // batch_size)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _default_records_dir(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "training_records"


def _chunks(items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
