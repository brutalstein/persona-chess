import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from persona_chess._version import __version__
from persona_chess.exceptions import ArtifactError
from persona_chess.profile.types import PersonaProfile

SCHEMA_VERSION = "persona-chess/v1"


@dataclass(frozen=True, slots=True)
class PersonaArtifact:
    schema_version: str
    package_version: str
    created_at: str
    model_type: str
    profile: PersonaProfile
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        model_type: str,
        profile: PersonaProfile,
        payload: dict[str, Any],
    ) -> "PersonaArtifact":
        return cls(
            schema_version=SCHEMA_VERSION,
            package_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            profile=profile,
            payload=payload,
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as exc:
            raise ArtifactError(f"Unable to save artifact: {output_path}") from exc

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["profile"] = self.profile.to_dict()
        return data

    @classmethod
    def load(cls, path: str | Path) -> "PersonaArtifact":
        input_path = Path(path)
        try:
            with input_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load artifact: {input_path}") from exc

        if data.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(f"Unsupported artifact schema: {data.get('schema_version')}")

        return cls(
            schema_version=data["schema_version"],
            package_version=data["package_version"],
            created_at=data["created_at"],
            model_type=data["model_type"],
            profile=PersonaProfile.from_dict(data["profile"]),
            payload=data["payload"],
        )
