import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from persona_chess._version import __version__
from persona_chess.exceptions import ArtifactError
from persona_chess.neural.config import LoraConfig, NeuralTrainingConfig, TransformerPolicyConfig

ADAPTER_MANIFEST_SCHEMA = "persona-chess/neural-adapter-manifest/v1"


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    schema_version: str
    package_version: str
    created_at: str
    base_model: str
    player: str
    transformer: TransformerPolicyConfig
    lora: LoraConfig
    training: NeuralTrainingConfig
    move_vocabulary_size: int
    position_vocabulary_size: int
    training_examples: int

    @classmethod
    def create(
        cls,
        *,
        base_model: str,
        player: str,
        transformer: TransformerPolicyConfig,
        lora: LoraConfig,
        training: NeuralTrainingConfig,
        move_vocabulary_size: int,
        position_vocabulary_size: int,
        training_examples: int,
    ) -> "AdapterManifest":
        return cls(
            schema_version=ADAPTER_MANIFEST_SCHEMA,
            package_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            base_model=base_model,
            player=player,
            transformer=transformer,
            lora=lora,
            training=training,
            move_vocabulary_size=move_vocabulary_size,
            position_vocabulary_size=position_vocabulary_size,
            training_examples=training_examples,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transformer"] = self.transformer.to_dict()
        data["lora"] = self.lora.to_dict()
        data["training"] = self.training.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterManifest":
        if data.get("schema_version") != ADAPTER_MANIFEST_SCHEMA:
            raise ArtifactError(
                f"Unsupported adapter manifest schema: {data.get('schema_version')}"
            )
        move_vocabulary_size = data.get("move_vocabulary_size", data.get("vocabulary_size"))
        if move_vocabulary_size is None:
            raise ArtifactError("Adapter manifest is missing move vocabulary size.")

        return cls(
            schema_version=data["schema_version"],
            package_version=data["package_version"],
            created_at=data["created_at"],
            base_model=data["base_model"],
            player=data["player"],
            transformer=TransformerPolicyConfig.from_dict(data["transformer"]),
            lora=LoraConfig.from_dict(data["lora"]),
            training=NeuralTrainingConfig.from_dict(data["training"]),
            move_vocabulary_size=int(move_vocabulary_size),
            position_vocabulary_size=int(data.get("position_vocabulary_size", 0)),
            training_examples=data["training_examples"],
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            output_path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(f"Unable to save adapter manifest: {output_path}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "AdapterManifest":
        input_path = Path(path)
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load adapter manifest: {input_path}") from exc
        return cls.from_dict(data)
