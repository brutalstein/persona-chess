from types import SimpleNamespace

import pytest

from persona_chess.models.types import MovePrediction
from persona_chess.neural import ensure_hf_base_model_cached, verify_hf_base_model_usable
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


def test_transformers_remote_model_compat_adds_missing_tied_weights_attr() -> None:
    class FakePretrainedModel:
        pass

    transformers = SimpleNamespace(PreTrainedModel=FakePretrainedModel)

    hf_base_module._patch_transformers_remote_model_compat(transformers)

    assert FakePretrainedModel.all_tied_weights_keys == {}


def test_verify_hf_base_model_usable_returns_preflight_move(monkeypatch) -> None:
    expected = MovePrediction(
        move_uci="e2e4",
        san="e4",
        score=1.0,
        reason="hf_base_policy",
    )

    def fake_predict(*args: object, **kwargs: object) -> list[MovePrediction]:
        return [expected]

    monkeypatch.setattr(hf_base_module, "predict_hf_base_moves", fake_predict)

    assert verify_hf_base_model_usable("Maxlegrec/ChessBot") == expected


def test_verify_hf_base_model_usable_rejects_empty_policy(monkeypatch) -> None:
    def fake_predict(*args: object, **kwargs: object) -> list[MovePrediction]:
        return []

    monkeypatch.setattr(hf_base_module, "predict_hf_base_moves", fake_predict)

    with pytest.raises(RuntimeError, match="did not return a legal move"):
        verify_hf_base_model_usable("Maxlegrec/ChessBot")
