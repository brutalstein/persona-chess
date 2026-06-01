from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TorchRuntimeInfo:
    torch_version: str
    cuda_build: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_capability: tuple[int, int] | None
    selected_device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def torch_runtime_info(torch: Any, *, requested_device: str | None = None) -> TorchRuntimeInfo:
    selected_device = resolve_torch_device(torch, requested_device=requested_device)
    cuda_device_name = None
    cuda_capability = None
    cuda_device_count = 0
    if torch.cuda.is_available():
        cuda_device_count = int(torch.cuda.device_count())
        index = _device_index(selected_device)
        cuda_device_name = str(torch.cuda.get_device_name(index))
        major, minor = torch.cuda.get_device_capability(index)
        cuda_capability = (int(major), int(minor))
    return TorchRuntimeInfo(
        torch_version=str(torch.__version__),
        cuda_build=getattr(torch.version, "cuda", None),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        cuda_capability=cuda_capability if cuda_capability is not None else None,
        selected_device=selected_device,
    )


def resolve_torch_device(torch: Any, *, requested_device: str | None = None) -> str:
    if requested_device:
        device = torch.device(requested_device)
        if device.type == "cuda":
            _require_usable_cuda(torch, requested_device=str(device))
        return str(device)
    if torch.cuda.is_available():
        _require_usable_cuda(torch, requested_device="cuda")
        return "cuda"
    return "cpu"


def cuda_diagnostic_message(torch: Any) -> str:
    version = str(torch.__version__)
    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        return (
            f"Installed torch={version} is a CPU-only build. For NVIDIA GPU training, "
            "install a CUDA-enabled PyTorch wheel. Current PyTorch stable wheels on PyPI "
            "target CUDA 13.x on supported platforms; see https://pytorch.org/get-started/locally/."
        )
    if not torch.cuda.is_available():
        return (
            f"Installed torch={version} was built with CUDA {cuda_build}, but no usable CUDA "
            "device is visible. Check your NVIDIA driver, GPU support, and environment variables "
            "such as CUDA_VISIBLE_DEVICES."
        )
    return f"CUDA is available through torch={version} built with CUDA {cuda_build}."


def _require_usable_cuda(torch: Any, *, requested_device: str) -> None:
    if getattr(torch.version, "cuda", None) is None:
        raise RuntimeError(cuda_diagnostic_message(torch))
    if not torch.cuda.is_available():
        raise RuntimeError(cuda_diagnostic_message(torch))
    index = _device_index(requested_device)
    if index >= int(torch.cuda.device_count()):
        raise RuntimeError(
            f"Requested CUDA device {requested_device!r}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )
    try:
        torch.cuda.get_device_properties(index)
    except Exception as exc:
        raise RuntimeError(
            f"CUDA device {requested_device!r} is visible but not usable: {exc}"
        ) from exc


def _device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return 0
