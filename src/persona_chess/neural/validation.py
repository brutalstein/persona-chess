from dataclasses import asdict, dataclass
from typing import Any

from persona_chess.neural.manifest import AdapterManifest
from persona_chess.neural.position_vocabulary import PositionVocabulary
from persona_chess.neural.vocabulary import MoveVocabulary


@dataclass(frozen=True, slots=True)
class NeuralArtifactValidation:
    ok: bool
    errors: tuple[str, ...]
    player: str
    move_vocabulary_size: int
    position_vocabulary_size: int
    training_examples: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["errors"] = list(self.errors)
        return data


def validate_neural_artifacts(
    *,
    manifest: AdapterManifest,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
) -> NeuralArtifactValidation:
    errors: list[str] = []

    if manifest.move_vocabulary_size != move_vocabulary.size:
        errors.append(
            "move_vocabulary_size mismatch: "
            f"manifest={manifest.move_vocabulary_size}, actual={move_vocabulary.size}"
        )

    if manifest.position_vocabulary_size != position_vocabulary.size:
        errors.append(
            "position_vocabulary_size mismatch: "
            f"manifest={manifest.position_vocabulary_size}, actual={position_vocabulary.size}"
        )

    if manifest.training_examples < 0:
        errors.append("training_examples must be non-negative")

    return NeuralArtifactValidation(
        ok=not errors,
        errors=tuple(errors),
        player=manifest.player,
        move_vocabulary_size=move_vocabulary.size,
        position_vocabulary_size=position_vocabulary.size,
        training_examples=manifest.training_examples,
    )
