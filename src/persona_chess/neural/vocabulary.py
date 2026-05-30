import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persona_chess.exceptions import ArtifactError
from persona_chess.training.records import TrainingRecord

MOVE_VOCABULARY_SCHEMA = "persona-chess/move-vocabulary/v1"
MOVE_PAD_TOKEN = "<move-pad>"
MOVE_UNK_TOKEN = "<move-unk>"


@dataclass(frozen=True, slots=True)
class MoveVocabulary:
    schema_version: str
    id_to_token: tuple[str, ...]

    @classmethod
    def from_records(cls, records: list[TrainingRecord]) -> "MoveVocabulary":
        moves = sorted({move for record in records for move in record.legal_moves})
        return cls(
            schema_version=MOVE_VOCABULARY_SCHEMA,
            id_to_token=(MOVE_PAD_TOKEN, MOVE_UNK_TOKEN, *moves),
        )

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, move_uci: str) -> int:
        return self.token_to_id().get(move_uci, self.unk_id)

    def decode(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.size:
            return MOVE_UNK_TOKEN
        return self.id_to_token[token_id]

    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id_to_token": list(self.id_to_token),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveVocabulary":
        if data.get("schema_version") != MOVE_VOCABULARY_SCHEMA:
            raise ArtifactError(f"Unsupported move vocabulary schema: {data.get('schema_version')}")
        return cls(
            schema_version=data["schema_version"],
            id_to_token=tuple(data["id_to_token"]),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            output_path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(f"Unable to save move vocabulary: {output_path}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "MoveVocabulary":
        input_path = Path(path)
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load move vocabulary: {input_path}") from exc
        return cls.from_dict(data)
