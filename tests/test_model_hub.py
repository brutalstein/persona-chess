import hashlib
import json
import zipfile
from pathlib import Path

from persona_chess.neural import ModelRegistry
from persona_chess.neural.model_hub import download_remote_model, resolve_model_reference


def test_model_registry_downloads_and_caches_zip_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(checkpoint_dir / "checkpoint.json", "checkpoint/checkpoint.json")

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "persona-chess/model-registry/v1",
                "models": [
                    {
                        "name": "persona-chess/base-test",
                        "version": "0.1.0",
                        "url": f"file://{archive}",
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry.load(registry_path)

    result = download_remote_model(
        registry.get("persona-chess/base-test"),
        cache_dir=tmp_path / "cache",
    )
    resolved = resolve_model_reference(
        "persona-chess/base-test",
        registry=registry,
        cache_dir=tmp_path / "cache",
    )

    assert result.downloaded
    assert (result.path / "checkpoint.json").exists()
    assert resolved == result.path
