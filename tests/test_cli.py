import json
from pathlib import Path

from typer.testing import CliRunner

from persona_chess.cli import app

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Train and inspect lightweight chess personas" in result.output


def test_cli_neural_move_help() -> None:
    result = CliRunner().invoke(app, ["neural-move", "--help"])

    assert result.exit_code == 0
    assert "Neural checkpoint directory" in result.output


def test_cli_engine_move_help() -> None:
    result = CliRunner().invoke(app, ["engine-move", "--help"])

    assert result.exit_code == 0
    assert "UCI engine binary" in result.output


def test_cli_streaming_training_and_neural_preparation(tmp_path: Path) -> None:
    runner = CliRunner()
    records = tmp_path / "target.train.jsonl"
    manifest = tmp_path / "adapter.manifest.json"
    move_vocab = tmp_path / "moves.vocab.json"
    position_vocab = tmp_path / "positions.vocab.json"

    export = runner.invoke(
        app,
        [
            "export-training-stream",
            str(FIXTURE),
            "Target Player",
            "--out",
            str(records),
        ],
    )
    prepare = runner.invoke(
        app,
        [
            "prepare-neural-stream",
            str(records),
            "Target Player",
            "--manifest-out",
            str(manifest),
            "--move-vocab-out",
            str(move_vocab),
            "--position-vocab-out",
            str(position_vocab),
            "--batch-size",
            "4",
            "--mixed-precision",
            "off",
            "--warmup-ratio",
            "0",
            "--max-grad-norm",
            "0.5",
        ],
    )

    assert export.exit_code == 0
    assert "Wrote 10 training records" in export.output
    assert prepare.exit_code == 0
    assert "Counted 10 streaming training records" in prepare.output
    training = json.loads(manifest.read_text(encoding="utf-8"))["training"]
    assert training["batch_size"] == 4
    assert training["mixed_precision"] == "off"
    assert training["warmup_ratio"] == 0.0
    assert training["max_grad_norm"] == 0.5
    assert manifest.exists()
    assert move_vocab.exists()
    assert position_vocab.exists()


def test_cli_exports_base_training_and_streaming_validation_split(tmp_path: Path) -> None:
    runner = CliRunner()
    base_records = tmp_path / "base.train.jsonl"
    train_records = tmp_path / "split.train.jsonl"
    validation_records = tmp_path / "split.valid.jsonl"

    export = runner.invoke(
        app,
        [
            "export-base-training-stream",
            str(FIXTURE),
            "--out",
            str(base_records),
        ],
    )
    split = runner.invoke(
        app,
        [
            "split-training-stream",
            str(base_records),
            "--train-out",
            str(train_records),
            "--validation-out",
            str(validation_records),
            "--validation-ratio",
            "0.5",
        ],
    )

    train_count = len(train_records.read_text(encoding="utf-8").splitlines())
    validation_count = len(validation_records.read_text(encoding="utf-8").splitlines())

    assert export.exit_code == 0
    assert "Wrote 20 base training records" in export.output
    assert split.exit_code == 0
    assert train_count + validation_count == 20
    assert '"validation_records"' in split.output


def test_cli_recommends_neural_config() -> None:
    result = CliRunner().invoke(
        app,
        [
            "recommend-neural-config",
            "--training-examples",
            "1000",
            "--config-profile",
            "small",
            "--device",
            "cpu",
        ],
    )

    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["profile"] == "small"
    assert payload["training"]["batch_size"] == 16
    assert payload["lora"]["rank"] == 4


def test_cli_model_card_writes_json_and_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    json_out = tmp_path / "target.model-card.json"
    markdown_out = tmp_path / "target.model-card.md"

    json_result = runner.invoke(
        app,
        [
            "model-card",
            str(FIXTURE),
            "Target Player",
            "--out",
            str(json_out),
        ],
    )
    markdown_result = runner.invoke(
        app,
        [
            "model-card",
            str(FIXTURE),
            "Target Player",
            "--out",
            str(markdown_out),
            "--format",
            "markdown",
        ],
    )

    assert json_result.exit_code == 0
    assert markdown_result.exit_code == 0
    assert '"schema_version": "persona-chess/model-card/v1"' in json_out.read_text(encoding="utf-8")
    assert "# Persona Model Card: Target Player" in markdown_out.read_text(encoding="utf-8")


def test_cli_prepare_and_validate_neural_artifacts(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest = tmp_path / "adapter.manifest.json"
    move_vocab = tmp_path / "moves.vocab.json"
    position_vocab = tmp_path / "positions.vocab.json"

    prepare = runner.invoke(
        app,
        [
            "prepare-neural",
            str(FIXTURE),
            "Target Player",
            "--manifest-out",
            str(manifest),
            "--move-vocab-out",
            str(move_vocab),
            "--position-vocab-out",
            str(position_vocab),
        ],
    )

    assert prepare.exit_code == 0
    assert manifest.exists()
    assert move_vocab.exists()
    assert position_vocab.exists()

    validate = runner.invoke(
        app,
        [
            "validate-neural",
            str(manifest),
            str(move_vocab),
            str(position_vocab),
        ],
    )

    assert validate.exit_code == 0
    assert '"ok": true' in validate.output
