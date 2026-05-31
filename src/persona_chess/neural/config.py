from dataclasses import asdict, dataclass
from typing import Any, Literal

MixedPrecisionMode = Literal["auto", "off", "fp16", "bf16"]


@dataclass(frozen=True, slots=True)
class TransformerPolicyConfig:
    max_sequence_length: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _require_positive("max_sequence_length", self.max_sequence_length)
        _require_positive("d_model", self.d_model)
        _require_positive("n_layers", self.n_layers)
        _require_positive("n_heads", self.n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        _require_probability("dropout", self.dropout)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerPolicyConfig":
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LoraConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("out_proj", "linear1", "linear2")

    def __post_init__(self) -> None:
        _require_positive("rank", self.rank)
        _require_positive("alpha", self.alpha)
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        _require_probability("dropout", self.dropout)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoraConfig":
        payload = dict(data)
        payload["target_modules"] = tuple(payload["target_modules"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class NeuralTrainingConfig:
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    max_grad_norm: float | None = 1.0
    mixed_precision: MixedPrecisionMode = "auto"
    seed: int = 42

    def __post_init__(self) -> None:
        _require_positive("epochs", self.epochs)
        _require_positive("batch_size", self.batch_size)
        _require_positive("gradient_accumulation_steps", self.gradient_accumulation_steps)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        _require_probability("warmup_ratio", self.warmup_ratio)
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when set")
        if self.mixed_precision not in {"auto", "off", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of auto, off, fp16, or bf16")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuralTrainingConfig":
        payload = dict(data)
        payload.setdefault("gradient_accumulation_steps", 1)
        payload.setdefault("max_grad_norm", 1.0)
        payload.setdefault("mixed_precision", "auto")
        return cls(**payload)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_probability(name: str, value: float) -> None:
    if not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1)")
