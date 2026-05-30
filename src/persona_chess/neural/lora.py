from dataclasses import asdict, dataclass
from typing import Any

from persona_chess.exceptions import OptionalDependencyError
from persona_chess.neural.config import LoraConfig


@dataclass(frozen=True, slots=True)
class LoraAdapterSummary:
    target_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int
    trainable_ratio: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data


def is_peft_available() -> bool:
    try:
        __import__("peft")
    except ModuleNotFoundError:
        return False
    return True


def require_peft() -> Any:
    try:
        import peft  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "PEFT is required for LoRA training. Install persona-chess with the ml extra."
        ) from exc
    return peft


def apply_lora_adapter(model: Any, config: LoraConfig) -> tuple[Any, LoraAdapterSummary]:
    peft = require_peft()
    _ensure_target_modules_exist(model, config.target_modules)

    peft_config = peft.LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    adapted_model = peft.get_peft_model(model, peft_config)
    summary = summarize_trainable_parameters(
        adapted_model,
        target_modules=config.target_modules,
    )
    return adapted_model, summary


def summarize_trainable_parameters(
    model: Any,
    *,
    target_modules: tuple[str, ...] = (),
) -> LoraAdapterSummary:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        parameter_count = int(parameter.numel())
        total += parameter_count
        if bool(parameter.requires_grad):
            trainable += parameter_count

    return LoraAdapterSummary(
        target_modules=target_modules,
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_ratio=trainable / total if total else 0.0,
    )


def _ensure_target_modules_exist(model: Any, target_modules: tuple[str, ...]) -> None:
    module_names = [name for name, _ in model.named_modules()]
    missing = [
        target_module
        for target_module in target_modules
        if not any(name.endswith(target_module) for name in module_names)
    ]
    if missing:
        raise ValueError(f"LoRA target modules not found: {', '.join(missing)}")
