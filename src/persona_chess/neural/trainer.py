from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from persona_chess.neural.config import LoraConfig, NeuralTrainingConfig, TransformerPolicyConfig
from persona_chess.neural.lora import apply_lora_adapter, summarize_trainable_parameters
from persona_chess.neural.samples import PolicyBatch
from persona_chess.neural.torch_backend import (
    build_transformer_policy_model,
    gather_legal_logits,
    policy_batch_to_tensors,
    require_torch,
)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    epochs: int
    steps: int
    final_loss: float
    trainable_parameters: int = 0
    total_parameters: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingResult":
        return cls(**data)


def train_policy_model(
    batches: list[PolicyBatch],
    *,
    transformer: TransformerPolicyConfig,
    training: NeuralTrainingConfig,
    position_vocabulary_size: int,
    move_vocabulary_size: int,
    device: str | None = None,
    lora: LoraConfig | None = None,
) -> tuple[Any, TrainingResult]:
    return train_policy_model_streaming(
        lambda: iter(batches),
        transformer=transformer,
        training=training,
        position_vocabulary_size=position_vocabulary_size,
        move_vocabulary_size=move_vocabulary_size,
        device=device,
        lora=lora,
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
) -> tuple[Any, TrainingResult]:
    torch = require_torch()
    torch.manual_seed(training.seed)

    model = build_transformer_policy_model(
        config=transformer,
        position_vocabulary_size=position_vocabulary_size,
        move_vocabulary_size=move_vocabulary_size,
    )
    if device:
        model = model.to(device)
    parameter_summary = summarize_trainable_parameters(model)

    if lora is not None:
        model, parameter_summary = apply_lora_adapter(model, lora)
        if device:
            model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    steps = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(training.epochs):
        accumulated = 0
        for batch in batch_factory():
            if batch.size == 0:
                continue
            tensors = policy_batch_to_tensors(batch, device=device)
            logits = model(tensors["input_ids"], tensors["attention_mask"])
            legal_logits = gather_legal_logits(
                logits,
                tensors["legal_move_ids"],
                tensors["legal_move_mask"],
            )
            loss = loss_fn(legal_logits, tensors["target_legal_indices"])
            scaled_loss = loss / training.gradient_accumulation_steps

            scaled_loss.backward()
            accumulated += 1
            if accumulated >= training.gradient_accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0

            final_loss = float(loss.detach().cpu().item())
            steps += 1
        if accumulated:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return model, TrainingResult(
        epochs=training.epochs,
        steps=steps,
        final_loss=final_loss,
        trainable_parameters=parameter_summary.trainable_parameters,
        total_parameters=parameter_summary.total_parameters,
    )
