from pathlib import Path

from typer.testing import CliRunner

from persona_chess.cli import app

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Train and inspect lightweight chess personas" in result.output


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
