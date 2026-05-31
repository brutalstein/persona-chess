import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from persona_chess.exceptions import ArtifactError

MODEL_REGISTRY_SCHEMA = "persona-chess/model-registry/v1"


@dataclass(frozen=True, slots=True)
class RemoteModel:
    name: str
    version: str
    url: str
    sha256: str | None = None
    description: str = ""
    archive_format: str = "zip"

    def __post_init__(self) -> None:
        if self.archive_format != "zip":
            raise ValueError("Only zip model archives are supported.")

    @property
    def cache_key(self) -> str:
        safe_name = self.name.replace("/", "__")
        return f"{safe_name}-{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteModel":
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            url=str(data["url"]),
            sha256=data.get("sha256"),
            description=str(data.get("description", "")),
            archive_format=str(data.get("archive_format", "zip")),
        )


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    schema_version: str
    models: tuple[RemoteModel, ...]

    def get(self, name: str) -> RemoteModel:
        matches = [model for model in self.models if model.name == name]
        if not matches:
            raise KeyError(name)
        return sorted(matches, key=lambda model: model.version)[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "models": [model.to_dict() for model in self.models],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRegistry":
        if data.get("schema_version") != MODEL_REGISTRY_SCHEMA:
            raise ArtifactError(f"Unsupported model registry schema: {data.get('schema_version')}")
        return cls(
            schema_version=MODEL_REGISTRY_SCHEMA,
            models=tuple(RemoteModel.from_dict(model) for model in data.get("models", ())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelRegistry":
        input_path = Path(path)
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load model registry: {input_path}") from exc
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class ModelDownloadResult:
    model: RemoteModel
    path: Path
    downloaded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "path": str(self.path),
            "downloaded": self.downloaded,
        }


def default_model_cache_dir() -> Path:
    override = os.environ.get("PERSONA_CHESS_MODEL_CACHE")
    if override:
        return Path(override)
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if root:
        return Path(root) / "persona-chess" / "models"
    return Path.home() / ".cache" / "persona-chess" / "models"


def download_remote_model(
    model: RemoteModel,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> ModelDownloadResult:
    target_root = Path(cache_dir) if cache_dir is not None else default_model_cache_dir()
    target_dir = target_root / model.cache_key
    if target_dir.exists() and not force:
        return ModelDownloadResult(model=model, path=target_dir, downloaded=False)

    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="persona-chess-model-") as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "model.zip"
        _download_file(model.url, archive_path)
        if model.sha256 is not None:
            _verify_sha256(archive_path, model.sha256)
        extracted = temp_path / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)
        checkpoint_dir = _find_checkpoint_dir(extracted)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(checkpoint_dir, target_dir)

    return ModelDownloadResult(model=model, path=target_dir, downloaded=True)


def resolve_model_reference(
    reference: str | Path,
    *,
    registry: ModelRegistry | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    path = Path(reference)
    if path.exists():
        return path
    if registry is None:
        raise ArtifactError(
            f"Model reference is not a local path and no registry was provided: {reference}"
        )
    model = registry.get(str(reference))
    return download_remote_model(model, cache_dir=cache_dir, force=force).path


def _download_file(url: str, output: Path) -> None:
    if url.startswith("file://"):
        shutil.copyfile(Path(url.removeprefix("file://")), output)
        return
    with urllib.request.urlopen(url) as response:  # noqa: S310
        output.write_bytes(response.read())


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.casefold() != expected.casefold():
        raise ArtifactError("Downloaded model checksum does not match registry sha256.")


def _find_checkpoint_dir(root: Path) -> Path:
    candidates = [path for path in root.rglob("checkpoint.json") if path.is_file()]
    if not candidates:
        raise ArtifactError("Model archive does not contain a checkpoint.json file.")
    return candidates[0].parent
