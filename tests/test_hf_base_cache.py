from types import SimpleNamespace

from persona_chess.neural import ensure_hf_base_model_cached
from persona_chess.neural import hf_base as hf_base_module


def test_ensure_hf_base_model_cached_uses_huggingface_snapshot(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_import(name: str) -> object:
        assert name == "huggingface_hub"

        def snapshot_download(*, repo_id: str) -> str:
            calls.append(repo_id)
            return "cache-path"

        return SimpleNamespace(snapshot_download=snapshot_download)

    monkeypatch.setattr(hf_base_module, "_require_module", fake_import)

    path = ensure_hf_base_model_cached("Maxlegrec/ChessBot")

    assert path == "cache-path"
    assert calls == ["Maxlegrec/ChessBot"]
