import subprocess
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
    nvidia_driver_version: str | None = None
    nvidia_smi_cuda_version: str | None = None

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
    nvidia = _nvidia_smi_versions()
    return TorchRuntimeInfo(
        torch_version=str(torch.__version__),
        cuda_build=getattr(torch.version, "cuda", None),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        cuda_capability=cuda_capability if cuda_capability is not None else None,
        selected_device=selected_device,
        nvidia_driver_version=nvidia.get("driver_version"),
        nvidia_smi_cuda_version=nvidia.get("cuda_version"),
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
        nvidia = _nvidia_smi_versions()
        detected = ""
        if nvidia:
            detected = (
                f" NVIDIA driver {nvidia.get('driver_version') or 'unknown'} reports "
                f"CUDA runtime support {nvidia.get('cuda_version') or 'unknown'}, so your GPU "
                "is visible to the system but not to this PyTorch build."
            )
        return (
            f"Installed torch={version} is a CPU-only build. For NVIDIA GPU training, "
            "install a CUDA-enabled PyTorch wheel selected from "
            "https://pytorch.org/get-started/locally/."
            f"{detected}"
        )
    if not torch.cuda.is_available():
        return (
            f"Installed torch={version} was built with CUDA {cuda_build}, but no usable CUDA "
            "device is visible. Check your NVIDIA driver, GPU support, and environment variables "
            "such as CUDA_VISIBLE_DEVICES."
        )
    return f"CUDA is available through torch={version} built with CUDA {cuda_build}."


def _nvidia_smi_versions() -> dict[str, str]:
    details: dict[str, str] = {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    first_line = completed.stdout.strip().splitlines()[0:1]
    if first_line:
        details["driver_version"] = first_line[0].strip()
    try:
        smi = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return details
    marker = "CUDA Version:"
    if marker in smi.stdout:
        tail = smi.stdout.split(marker, 1)[1]
        details["cuda_version"] = tail.split("|", 1)[0].strip()
    return details


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
