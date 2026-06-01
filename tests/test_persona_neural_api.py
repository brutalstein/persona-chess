from pathlib import Path

import pytest

from persona_chess import PersonaChess
from persona_chess.neural import is_torch_available

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_persona_train_builds_neural_checkpoint(tmp_path: Path) -> None:
    if not is_torch_available():
        pytest.skip("PyTorch is not installed in this environment.")

    persona = PersonaChess()
    result = persona.train(
        FIXTURE,
        player="Target Player",
        output_dir=tmp_path,
        use_lora=False,
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        show_progress=False,
        prefetch_base_model=False,
        mixed_precision="off",
    )

    assert persona.neural_checkpoint_dir == result.checkpoint_dir
    assert result.checkpoint_dir.parent == tmp_path
    assert result.model_state_path.exists()
    assert result.model_state_path.name == "model.pt"
    assert result.adapter_manifest.base_model == "Maxlegrec/ChessBot"
    assert result.training_examples > 0
    assert result.training_result.optimizer_steps > 0

    bot = PersonaChess.load_neural(result.checkpoint_dir)
    move = bot.move("startpos", device="cpu", use_base_model=False)
    predictions = bot.predict_neural("startpos", top_k=2, device="cpu", use_base_model=False)

    assert move.reason == "neural_policy"
    assert predictions
    assert all(prediction.reason == "neural_policy" for prediction in predictions)


def test_persona_move_can_blend_hf_base_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not is_torch_available():
        pytest.skip("PyTorch is not installed in this environment.")

    persona = PersonaChess()
    result = persona.train(
        FIXTURE,
        player="Target Player",
        output_dir=tmp_path,
        use_lora=False,
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        show_progress=False,
        prefetch_base_model=False,
        mixed_precision="off",
    )

    def fake_base_moves(*args: object, **kwargs: object) -> list[object]:
        from persona_chess.models.types import MovePrediction

        return [
            MovePrediction(move_uci="e2e4", san="e4", score=0.9, reason="hf_base_policy"),
            MovePrediction(move_uci="d2d4", san="d4", score=0.1, reason="hf_base_policy"),
        ]

    monkeypatch.setattr("persona_chess.neural.inference.predict_hf_base_moves", fake_base_moves)
    bot = PersonaChess.load_neural(result.checkpoint_dir)
    move = bot.move("startpos", device="cpu")

    assert move.reason == "hf_base_persona_policy"


def test_persona_train_can_stream_from_pgn(tmp_path: Path) -> None:
    if not is_torch_available():
        pytest.skip("PyTorch is not installed in this environment.")

    persona = PersonaChess()
    result = persona.train(
        FIXTURE,
        player="Target Player",
        output_dir=tmp_path,
        streaming=True,
        use_lora=False,
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        show_progress=False,
        prefetch_base_model=False,
        mixed_precision="off",
    )

    assert result.training_records is not None
    assert result.training_records.exists()
    assert result.model_state_path.exists()
