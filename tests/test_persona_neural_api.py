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
        checkpoint_dir=tmp_path / "checkpoint",
        use_lora=False,
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        warmup_ratio=0.0,
        mixed_precision="off",
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )

    assert persona.neural_checkpoint_dir == result.checkpoint_dir
    assert result.model_state_path.exists()
    assert result.training_examples > 0
    assert result.training_result.optimizer_steps > 0

    predictions = persona.predict_neural("startpos", top_k=2, device="cpu")

    assert predictions
    assert all(prediction.reason == "neural_policy" for prediction in predictions)


def test_persona_train_records_uses_streaming_jsonl(tmp_path: Path) -> None:
    if not is_torch_available():
        pytest.skip("PyTorch is not installed in this environment.")

    persona = PersonaChess()
    records = tmp_path / "target.records.jsonl"
    written = persona.export_training_records(FIXTURE, records, player="Target Player")

    result = persona.train_records(
        records,
        player="Target Player",
        checkpoint_dir=tmp_path / "stream-checkpoint",
        use_lora=False,
        device="cpu",
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        warmup_ratio=0.0,
        mixed_precision="off",
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )

    assert written == result.training_examples
    assert result.training_records == records
    assert result.model_state_path.exists()
