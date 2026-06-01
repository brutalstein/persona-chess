from types import SimpleNamespace

import pytest

from persona_chess.neural import cuda as cuda_module
from persona_chess.neural import cuda_diagnostic_message, resolve_torch_device


class _FakeCuda:
    def __init__(self, *, available: bool, count: int = 0) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_properties(self, index: int) -> object:
        if index >= self._count:
            raise RuntimeError("bad device")
        return object()


class _FakeTorch:
    __version__ = "2.12.0+cpu"
    version = SimpleNamespace(cuda=None)
    cuda = _FakeCuda(available=False)

    @staticmethod
    def device(value: str) -> SimpleNamespace:
        device_type = value.split(":", 1)[0]
        return SimpleNamespace(type=device_type, __str__=lambda: value)


def test_cuda_diagnostic_reports_cpu_torch_with_visible_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cuda_module,
        "_nvidia_smi_versions",
        lambda: {"driver_version": "596.36", "cuda_version": "13.2"},
    )

    message = cuda_diagnostic_message(_FakeTorch)

    assert "CPU-only build" in message
    assert "NVIDIA driver 596.36" in message
    assert "CUDA runtime support 13.2" in message


def test_resolve_torch_device_rejects_cpu_torch_when_cuda_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cuda_module,
        "_nvidia_smi_versions",
        lambda: {"driver_version": "596.36", "cuda_version": "13.2"},
    )

    with pytest.raises(RuntimeError, match="CPU-only build"):
        resolve_torch_device(_FakeTorch, requested_device="cuda")
