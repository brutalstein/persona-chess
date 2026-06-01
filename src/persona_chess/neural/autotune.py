import ctypes
import math
import os
from dataclasses import asdict, dataclass, replace
from sys import platform
from typing import Any, ClassVar, Literal, TypeAlias

from persona_chess.neural.config import LoraConfig, NeuralTrainingConfig, TransformerPolicyConfig
from persona_chess.neural.cuda import resolve_torch_device
from persona_chess.neural.torch_backend import is_torch_available, require_torch

NeuralConfigProfile: TypeAlias = Literal["auto", "small", "balanced", "large"]
ResolvedNeuralConfigProfile: TypeAlias = Literal["small", "balanced", "large"]


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    cpu_count: int
    memory_gb: float | None
    torch_available: bool
    cuda_available: bool
    cuda_device_name: str | None
    cuda_memory_gb: float | None
    device_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeuralAutoConfig:
    profile: ResolvedNeuralConfigProfile
    hardware: HardwareProfile
    transformer: TransformerPolicyConfig
    training: NeuralTrainingConfig
    lora: LoraConfig
    effective_batch_size: int
    notes: tuple[str, ...]

    def with_configs(
        self,
        *,
        transformer: TransformerPolicyConfig | None = None,
        training: NeuralTrainingConfig | None = None,
        lora: LoraConfig | None = None,
        notes: tuple[str, ...] = (),
    ) -> "NeuralAutoConfig":
        resolved_training = training or self.training
        return replace(
            self,
            transformer=transformer or self.transformer,
            training=resolved_training,
            lora=lora or self.lora,
            effective_batch_size=(
                resolved_training.batch_size * resolved_training.gradient_accumulation_steps
            ),
            notes=self.notes + notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "hardware": self.hardware.to_dict(),
            "transformer": self.transformer.to_dict(),
            "training": self.training.to_dict(),
            "lora": self.lora.to_dict(),
            "effective_batch_size": self.effective_batch_size,
            "notes": list(self.notes),
        }


def detect_hardware_profile(*, device: str | None = None) -> HardwareProfile:
    torch_available = is_torch_available()
    cuda_available = False
    cuda_device_name: str | None = None
    cuda_memory_gb: float | None = None
    normalized_device = (device or "").lower()
    forced_cpu = normalized_device == "cpu"

    if torch_available and not forced_cpu:
        torch = require_torch()
        try:
            resolved_device = resolve_torch_device(torch, requested_device=device)
        except RuntimeError:
            resolved_device = "cpu"
        if resolved_device.startswith("cuda"):
            device_index = _parse_cuda_device_index(resolved_device)
            properties = torch.cuda.get_device_properties(device_index)
            cuda_available = True
            cuda_device_name = str(properties.name)
            cuda_memory_gb = _bytes_to_gb(int(properties.total_memory))

    return HardwareProfile(
        cpu_count=os.cpu_count() or 1,
        memory_gb=_detect_system_memory_gb(),
        torch_available=torch_available,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        cuda_memory_gb=cuda_memory_gb,
        device_type="cuda" if cuda_available else "cpu",
    )


def recommend_neural_config(
    training_examples: int,
    *,
    profile: NeuralConfigProfile = "auto",
    hardware: HardwareProfile | None = None,
    device: str | None = None,
) -> NeuralAutoConfig:
    if training_examples <= 0:
        raise ValueError("training_examples must be positive")

    resolved_hardware = hardware or detect_hardware_profile(device=device)
    resolved_profile = (
        _select_auto_profile(training_examples, resolved_hardware) if profile == "auto" else profile
    )
    transformer = _transformer_for_profile(resolved_profile)
    batch_size = _micro_batch_for_profile(resolved_profile, resolved_hardware)
    effective_batch_size = _effective_batch_for_profile(resolved_profile, training_examples)
    accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
    training = NeuralTrainingConfig(
        epochs=_epochs_for_dataset(training_examples),
        batch_size=batch_size,
        learning_rate=_learning_rate_for_profile(resolved_profile),
        gradient_accumulation_steps=accumulation_steps,
        warmup_ratio=_warmup_ratio_for_dataset(training_examples),
    )
    lora = _lora_for_profile(resolved_profile)

    return NeuralAutoConfig(
        profile=resolved_profile,
        hardware=resolved_hardware,
        transformer=transformer,
        training=training,
        lora=lora,
        effective_batch_size=training.batch_size * training.gradient_accumulation_steps,
        notes=_build_notes(
            requested_profile=profile,
            resolved_profile=resolved_profile,
            training_examples=training_examples,
            hardware=resolved_hardware,
            training=training,
        ),
    )


def _select_auto_profile(
    training_examples: int, hardware: HardwareProfile
) -> ResolvedNeuralConfigProfile:
    if hardware.cuda_available:
        cuda_memory = hardware.cuda_memory_gb or 0.0
        if cuda_memory >= 16 and training_examples >= 50_000:
            return "large"
        if cuda_memory >= 8:
            return "balanced"
        return "small"

    memory = hardware.memory_gb or 0.0
    if memory >= 64 and hardware.cpu_count >= 16 and training_examples <= 250_000:
        return "balanced"
    return "small"


def _transformer_for_profile(profile: ResolvedNeuralConfigProfile) -> TransformerPolicyConfig:
    if profile == "large":
        return TransformerPolicyConfig(d_model=384, n_layers=6, n_heads=8, dropout=0.1)
    if profile == "balanced":
        return TransformerPolicyConfig(d_model=256, n_layers=4, n_heads=8, dropout=0.1)
    return TransformerPolicyConfig(d_model=128, n_layers=2, n_heads=4, dropout=0.1)


def _micro_batch_for_profile(
    profile: ResolvedNeuralConfigProfile, hardware: HardwareProfile
) -> int:
    if not hardware.cuda_available:
        return 16 if profile == "small" else 24

    cuda_memory = hardware.cuda_memory_gb or 0.0
    if profile == "large":
        return 96 if cuda_memory >= 24 else 64
    if profile == "balanced":
        return 64 if cuda_memory >= 10 else 32
    return 32 if cuda_memory >= 6 else 16


def _effective_batch_for_profile(
    profile: ResolvedNeuralConfigProfile, training_examples: int
) -> int:
    if training_examples < 1_000:
        return 32
    if profile == "large":
        return 256
    if profile == "balanced":
        return 128
    return 64


def _epochs_for_dataset(training_examples: int) -> int:
    if training_examples < 500:
        return 12
    if training_examples < 5_000:
        return 8
    if training_examples < 50_000:
        return 5
    if training_examples < 1_000_000:
        return 3
    if training_examples < 10_000_000:
        return 2
    return 1


def _learning_rate_for_profile(profile: ResolvedNeuralConfigProfile) -> float:
    if profile == "large":
        return 2e-4
    return 3e-4


def _warmup_ratio_for_dataset(training_examples: int) -> float:
    return 0.03 if training_examples >= 1_000_000 else 0.05


def _lora_for_profile(profile: ResolvedNeuralConfigProfile) -> LoraConfig:
    if profile == "large":
        return LoraConfig(rank=16, alpha=32, dropout=0.05)
    if profile == "balanced":
        return LoraConfig(rank=8, alpha=16, dropout=0.05)
    return LoraConfig(rank=4, alpha=8, dropout=0.05)


def _build_notes(
    *,
    requested_profile: NeuralConfigProfile,
    resolved_profile: ResolvedNeuralConfigProfile,
    training_examples: int,
    hardware: HardwareProfile,
    training: NeuralTrainingConfig,
) -> tuple[str, ...]:
    notes = [
        f"Selected {resolved_profile!r} profile for {training_examples} training examples.",
        (
            "Using CUDA-aware batch sizing."
            if hardware.cuda_available
            else "Using conservative CPU-friendly batch sizing."
        ),
        (
            "Gradient accumulation keeps the effective batch stable without forcing a large "
            "micro-batch into memory."
        ),
    ]
    if requested_profile != "auto":
        notes.append(f"Profile was explicitly requested as {requested_profile!r}.")
    if training.epochs == 1:
        notes.append(
            "Large datasets default to one full pass; increase epochs for extra adaptation."
        )
    return tuple(notes)


def _parse_cuda_device_index(device: str) -> int:
    if not device.startswith("cuda:"):
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return 0


def _detect_system_memory_gb() -> float | None:
    if platform == "win32":
        return _detect_windows_memory_gb()
    return _detect_posix_memory_gb()


def _detect_windows_memory_gb() -> float | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    try:
        if not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except OSError:
        return None
    return _bytes_to_gb(int(status.ullTotalPhys))


def _detect_posix_memory_gb() -> float | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    return _bytes_to_gb(pages * page_size)


def _bytes_to_gb(value: int) -> float:
    return round(value / (1024**3), 2)
