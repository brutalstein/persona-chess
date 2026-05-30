import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from persona_chess._version import __version__
from persona_chess.exceptions import ArtifactError
from persona_chess.neural.lora import apply_lora_adapter
from persona_chess.neural.manifest import AdapterManifest
from persona_chess.neural.position_vocabulary import PositionVocabulary
from persona_chess.neural.torch_backend import build_transformer_policy_model, require_torch
from persona_chess.neural.trainer import TrainingResult
from persona_chess.neural.validation import validate_neural_artifacts
from persona_chess.neural.vocabulary import MoveVocabulary

NEURAL_CHECKPOINT_SCHEMA = "persona-chess/neural-checkpoint/v1"
CHECKPOINT_MANIFEST_FILE = "checkpoint.json"
ADAPTER_MANIFEST_FILE = "adapter.manifest.json"
MOVE_VOCABULARY_FILE = "moves.vocab.json"
POSITION_VOCABULARY_FILE = "positions.vocab.json"
MODEL_STATE_FILE = "model.pt"


@dataclass(frozen=True, slots=True)
class NeuralCheckpointManifest:
    schema_version: str
    package_version: str
    created_at: str
    player: str
    model_state_file: str
    adapter_manifest_file: str
    move_vocabulary_file: str
    position_vocabulary_file: str
    training_result: TrainingResult | None
    lora_applied: bool

    @classmethod
    def create(
        cls,
        *,
        player: str,
        training_result: TrainingResult | None = None,
        model_state_file: str = MODEL_STATE_FILE,
        adapter_manifest_file: str = ADAPTER_MANIFEST_FILE,
        move_vocabulary_file: str = MOVE_VOCABULARY_FILE,
        position_vocabulary_file: str = POSITION_VOCABULARY_FILE,
        lora_applied: bool = False,
    ) -> "NeuralCheckpointManifest":
        return cls(
            schema_version=NEURAL_CHECKPOINT_SCHEMA,
            package_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            player=player,
            model_state_file=model_state_file,
            adapter_manifest_file=adapter_manifest_file,
            move_vocabulary_file=move_vocabulary_file,
            position_vocabulary_file=position_vocabulary_file,
            training_result=training_result,
            lora_applied=lora_applied,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["training_result"] = (
            self.training_result.to_dict() if self.training_result is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuralCheckpointManifest":
        if data.get("schema_version") != NEURAL_CHECKPOINT_SCHEMA:
            raise ArtifactError(f"Unsupported checkpoint schema: {data.get('schema_version')}")

        training_result_data = data.get("training_result")
        training_result = (
            TrainingResult.from_dict(training_result_data)
            if training_result_data is not None
            else None
        )
        return cls(
            schema_version=data["schema_version"],
            package_version=data["package_version"],
            created_at=data["created_at"],
            player=data["player"],
            model_state_file=data["model_state_file"],
            adapter_manifest_file=data["adapter_manifest_file"],
            move_vocabulary_file=data["move_vocabulary_file"],
            position_vocabulary_file=data["position_vocabulary_file"],
            training_result=training_result,
            lora_applied=bool(data.get("lora_applied", False)),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            output_path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(f"Unable to save checkpoint manifest: {output_path}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "NeuralCheckpointManifest":
        input_path = Path(path)
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load checkpoint manifest: {input_path}") from exc
        return cls.from_dict(data)


def save_torch_policy_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    adapter_manifest: AdapterManifest,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
    training_result: TrainingResult | None = None,
    lora_applied: bool = False,
) -> NeuralCheckpointManifest:
    torch = require_torch()
    checkpoint_dir = Path(directory)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_neural_artifacts(
        manifest=adapter_manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
    )
    if not validation.ok:
        raise ArtifactError("; ".join(validation.errors))

    checkpoint_manifest = NeuralCheckpointManifest.create(
        player=adapter_manifest.player,
        training_result=training_result,
        lora_applied=lora_applied,
    )

    adapter_manifest.save(checkpoint_dir / checkpoint_manifest.adapter_manifest_file)
    move_vocabulary.save(checkpoint_dir / checkpoint_manifest.move_vocabulary_file)
    position_vocabulary.save(checkpoint_dir / checkpoint_manifest.position_vocabulary_file)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_dir / MODEL_STATE_FILE)
    checkpoint_manifest.save(checkpoint_dir / CHECKPOINT_MANIFEST_FILE)
    return checkpoint_manifest


def load_torch_policy_checkpoint(
    directory: str | Path, *, device: str | None = None
) -> tuple[Any, ...]:
    torch = require_torch()
    checkpoint_dir = Path(directory)
    checkpoint_manifest = NeuralCheckpointManifest.load(checkpoint_dir / CHECKPOINT_MANIFEST_FILE)
    adapter_manifest = AdapterManifest.load(
        checkpoint_dir / checkpoint_manifest.adapter_manifest_file
    )
    move_vocabulary = MoveVocabulary.load(checkpoint_dir / checkpoint_manifest.move_vocabulary_file)
    position_vocabulary = PositionVocabulary.load(
        checkpoint_dir / checkpoint_manifest.position_vocabulary_file
    )

    model = build_transformer_policy_model(
        config=adapter_manifest.transformer,
        position_vocabulary_size=position_vocabulary.size,
        move_vocabulary_size=move_vocabulary.size,
        pad_token_id=position_vocabulary.pad_id,
    )
    if checkpoint_manifest.lora_applied:
        model, _ = apply_lora_adapter(model, adapter_manifest.lora)

    state = torch.load(
        checkpoint_dir / checkpoint_manifest.model_state_file,
        map_location=torch.device(device) if device else None,
    )
    model.load_state_dict(state["model_state_dict"])
    if device:
        model = model.to(device)
    model.eval()
    return model, checkpoint_manifest, adapter_manifest, move_vocabulary, position_vocabulary
