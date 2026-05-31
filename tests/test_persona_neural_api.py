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
        mixed_precision="off",
    )

    assert persona.neural_checkpoint_dir == result.checkpoint_dir
    assert result.checkpoint_dir.parent == tmp_path
    assert result.model_state_path.exists()
    assert result.model_state_path.name == "model.pt"
    assert result.training_examples > 0
    assert result.training_result.optimizer_steps > 0

    predictions = persona.predict_neural("startpos", top_k=2, device="cpu")

    assert predictions
    assert all(prediction.reason == "neural_policy" for prediction in predictions)


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
        mixed_precision="off",
    )

    assert result.training_records is not None
    assert result.training_records.exists()
    assert result.model_state_path.exists()
