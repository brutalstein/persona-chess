import math
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

from persona_chess.neural.config import (
    LoraConfig,
    MixedPrecisionMode,
    NeuralTrainingConfig,
    TransformerPolicyConfig,
)
from persona_chess.neural.lora import apply_lora_adapter, summarize_trainable_parameters
from persona_chess.neural.samples import PolicyBatch
from persona_chess.neural.torch_backend import (
    build_transformer_policy_model,
    gather_legal_logits,
    policy_batch_to_tensors,
    require_torch,
)


@dataclass(frozen=True, slots=True)
class PolicyEvaluationResult:
    loss: float
    accuracy: float
    top3_accuracy: float
    examples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    epochs: int
    steps: int
    final_loss: float
    optimizer_steps: int = 0
    average_train_loss: float = 0.0
    validation_loss: float | None = None
    validation_accuracy: float | None = None
    validation_top3_accuracy: float | None = None
    validation_examples: int = 0
    best_validation_loss: float | None = None
    best_epoch: int | None = None
    mixed_precision: str = "off"
    trainable_parameters: int = 0
    total_parameters: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingResult":
        payload = dict(data)
        payload.setdefault("optimizer_steps", 0)
        payload.setdefault("average_train_loss", payload.get("final_loss", 0.0))
        payload.setdefault("validation_loss", None)
        payload.setdefault("validation_accuracy", None)
        payload.setdefault("validation_top3_accuracy", None)
        payload.setdefault("validation_examples", 0)
        payload.setdefault("best_validation_loss", payload.get("validation_loss"))
        payload.setdefault("best_epoch", None)
        payload.setdefault("mixed_precision", "off")
        payload.setdefault("trainable_parameters", 0)
        payload.setdefault("total_parameters", 0)
        return cls(**payload)


def train_policy_model(
    batches: list[PolicyBatch],
    *,
    transformer: TransformerPolicyConfig,
    training: NeuralTrainingConfig,
    position_vocabulary_size: int,
    move_vocabulary_size: int,
    device: str | None = None,
    lora: LoraConfig | None = None,
    validation_batches: list[PolicyBatch] | None = None,
) -> tuple[Any, TrainingResult]:
    return train_policy_model_streaming(
        lambda: iter(batches),
        transformer=transformer,
        training=training,
        position_vocabulary_size=position_vocabulary_size,
        move_vocabulary_size=move_vocabulary_size,
        device=device,
        lora=lora,
        validation_batch_factory=(lambda: iter(validation_batches))
        if validation_batches is not None
        else None,
        training_batches=len(batches),
    )


def train_policy_model_streaming(
    batch_factory: Callable[[], Iterable[PolicyBatch]],
    *,
    transformer: TransformerPolicyConfig,
    training: NeuralTrainingConfig,
    position_vocabulary_size: int,
    move_vocabulary_size: int,
    device: str | None = None,
    lora: LoraConfig | None = None,
    validation_batch_factory: Callable[[], Iterable[PolicyBatch]] | None = None,
    training_batches: int | None = None,
) -> tuple[Any, TrainingResult]:
    torch = require_torch()
    torch.manual_seed(training.seed)
    active_device = _resolve_device(torch, device)

    model = build_transformer_policy_model(
        config=transformer,
        position_vocabulary_size=position_vocabulary_size,
        move_vocabulary_size=move_vocabulary_size,
    )
    model = model.to(active_device)
    parameter_summary = summarize_trainable_parameters(model)

    if lora is not None:
        model, parameter_summary = apply_lora_adapter(model, lora)
        model = model.to(active_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = _build_scheduler(
        torch,
        optimizer,
        training=training,
        total_optimizer_steps=_estimate_optimizer_steps(
            training=training,
            training_batches=training_batches,
        ),
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    precision_plan = _resolve_precision_plan(torch, training.mixed_precision, active_device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision_plan.use_scaler)
    final_loss = 0.0
    steps = 0
    optimizer_steps = 0
    average_train_loss = 0.0
    validation_result: PolicyEvaluationResult | None = None
    best_validation_loss: float | None = None
    best_epoch: int | None = None

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, training.epochs + 1):
        accumulated = 0
        epoch_loss_sum = 0.0
        epoch_examples = 0
        for batch in batch_factory():
            if batch.size == 0:
                continue
            tensors = policy_batch_to_tensors(batch, device=str(active_device))
            with _autocast_context(torch, precision_plan, active_device):
                logits = model(tensors["input_ids"], tensors["attention_mask"])
                legal_logits = gather_legal_logits(
                    logits,
                    tensors["legal_move_ids"],
                    tensors["legal_move_mask"],
                )
                loss = loss_fn(legal_logits, tensors["target_legal_indices"])
            scaled_loss = loss / training.gradient_accumulation_steps

            if precision_plan.use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulated += 1
            if accumulated >= training.gradient_accumulation_steps:
                _optimizer_step(
                    torch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    max_grad_norm=training.max_grad_norm,
                )
                optimizer_steps += 1
                accumulated = 0

            final_loss = float(loss.detach().cpu().item())
            epoch_loss_sum += final_loss * batch.size
            epoch_examples += batch.size
            steps += 1
        if accumulated:
            _optimizer_step(
                torch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                max_grad_norm=training.max_grad_norm,
            )
            optimizer_steps += 1

        if epoch_examples:
            average_train_loss = epoch_loss_sum / epoch_examples

        if validation_batch_factory is not None:
            validation_result = evaluate_policy_model(
                model,
                validation_batch_factory(),
                device=str(active_device),
            )
            if best_validation_loss is None or validation_result.loss < best_validation_loss:
                best_validation_loss = validation_result.loss
                best_epoch = epoch
            model.train()

    return model, TrainingResult(
        epochs=training.epochs,
        steps=steps,
        final_loss=final_loss,
        optimizer_steps=optimizer_steps,
        average_train_loss=average_train_loss,
        validation_loss=validation_result.loss if validation_result is not None else None,
        validation_accuracy=(validation_result.accuracy if validation_result is not None else None),
        validation_top3_accuracy=(
            validation_result.top3_accuracy if validation_result is not None else None
        ),
        validation_examples=validation_result.examples if validation_result is not None else 0,
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        mixed_precision=precision_plan.label,
        trainable_parameters=parameter_summary.trainable_parameters,
        total_parameters=parameter_summary.total_parameters,
    )


def evaluate_policy_model(
    model: Any,
    batches: Iterable[PolicyBatch],
    *,
    device: str | None = None,
) -> PolicyEvaluationResult:
    torch = require_torch()
    loss_fn = torch.nn.CrossEntropyLoss()
    was_training = bool(model.training)
    model.eval()

    loss_sum = 0.0
    examples = 0
    correct = 0
    top3_correct = 0
    with torch.no_grad():
        for batch in batches:
            if batch.size == 0:
                continue
            tensors = policy_batch_to_tensors(batch, device=device)
            logits = model(tensors["input_ids"], tensors["attention_mask"])
            legal_logits = gather_legal_logits(
                logits,
                tensors["legal_move_ids"],
                tensors["legal_move_mask"],
            )
            targets = tensors["target_legal_indices"]
            loss = loss_fn(legal_logits, targets)
            loss_sum += float(loss.detach().cpu().item()) * batch.size
            examples += batch.size
            correct += int((legal_logits.argmax(dim=1) == targets).sum().detach().cpu().item())
            k = min(3, int(legal_logits.shape[1]))
            topk = legal_logits.topk(k=k, dim=1).indices
            top3_correct += int(
                (topk == targets.unsqueeze(1)).any(dim=1).sum().detach().cpu().item()
            )

    if was_training:
        model.train()

    if examples == 0:
        return PolicyEvaluationResult(loss=0.0, accuracy=0.0, top3_accuracy=0.0, examples=0)
    return PolicyEvaluationResult(
        loss=loss_sum / examples,
        accuracy=correct / examples,
        top3_accuracy=top3_correct / examples,
        examples=examples,
    )


@dataclass(frozen=True, slots=True)
class _PrecisionPlan:
    label: str
    enabled: bool
    dtype: Any | None
    use_scaler: bool


def _resolve_device(torch: Any, device: str | None) -> Any:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_precision_plan(
    torch: Any,
    mode: MixedPrecisionMode,
    device: Any,
) -> _PrecisionPlan:
    if mode == "off" or device.type != "cuda":
        return _PrecisionPlan(label="off", enabled=False, dtype=None, use_scaler=False)
    if mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return _PrecisionPlan(
                label="bf16", enabled=True, dtype=torch.bfloat16, use_scaler=False
            )
        return _PrecisionPlan(label="off", enabled=False, dtype=None, use_scaler=False)
    if mode == "fp16":
        return _PrecisionPlan(label="fp16", enabled=True, dtype=torch.float16, use_scaler=True)
    if torch.cuda.is_bf16_supported():
        return _PrecisionPlan(label="bf16", enabled=True, dtype=torch.bfloat16, use_scaler=False)
    return _PrecisionPlan(label="fp16", enabled=True, dtype=torch.float16, use_scaler=True)


def _autocast_context(torch: Any, precision: _PrecisionPlan, device: Any) -> Any:
    if not precision.enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.dtype)


def _estimate_optimizer_steps(
    *,
    training: NeuralTrainingConfig,
    training_batches: int | None,
) -> int | None:
    if training_batches is None or training_batches <= 0:
        return None
    total_batches = training_batches * training.epochs
    return max(1, math.ceil(total_batches / training.gradient_accumulation_steps))


def _build_scheduler(
    torch: Any,
    optimizer: Any,
    *,
    training: NeuralTrainingConfig,
    total_optimizer_steps: int | None,
) -> Any | None:
    if total_optimizer_steps is None:
        return None

    warmup_steps = int(total_optimizer_steps * training.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        remaining_steps = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(remaining_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _optimizer_step(
    torch: Any,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    scaler: Any,
    max_grad_norm: float | None,
) -> None:
    if max_grad_norm is not None:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
