import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from persona_chess._version import __version__
from persona_chess.exceptions import ArtifactError
from persona_chess.profile.types import PersonaProfile

MODEL_CARD_SCHEMA = "persona-chess/model-card/v1"


@dataclass(frozen=True, slots=True)
class DataQualitySummary:
    games: int
    examples: int
    unique_positions: int
    duplicate_position_rate: float
    average_legal_moves: float
    confidence: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataQualitySummary":
        payload = dict(data)
        payload["warnings"] = tuple(payload["warnings"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class StyleSummary:
    tags: tuple[str, ...]
    forcing_rate: float
    capture_rate: float
    check_rate: float
    castle_rate: float
    early_queen_rate: float
    opening_phase_rate: float
    average_fullmove_number: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StyleSummary":
        payload = dict(data)
        payload["tags"] = tuple(payload["tags"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class MoveBreakdown:
    phase_distribution: dict[str, int]
    piece_distribution: dict[str, int]
    piece_rates: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveBreakdown":
        return cls(
            phase_distribution=dict(data["phase_distribution"]),
            piece_distribution=dict(data["piece_distribution"]),
            piece_rates=dict(data["piece_rates"]),
        )


@dataclass(frozen=True, slots=True)
class ModelCardRecommendation:
    recommended_model: str
    recommended_inference: str
    neural_readiness: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCardRecommendation":
        payload = dict(data)
        payload["notes"] = tuple(payload["notes"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PersonaModelCard:
    schema_version: str
    package_version: str
    created_at: str
    player: str
    profile: PersonaProfile
    data_quality: DataQualitySummary
    style: StyleSummary
    move_breakdown: MoveBreakdown
    recommendation: ModelCardRecommendation

    @classmethod
    def create(
        cls,
        *,
        profile: PersonaProfile,
        data_quality: DataQualitySummary,
        style: StyleSummary,
        move_breakdown: MoveBreakdown,
        recommendation: ModelCardRecommendation,
    ) -> "PersonaModelCard":
        return cls(
            schema_version=MODEL_CARD_SCHEMA,
            package_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            player=profile.player,
            profile=profile,
            data_quality=data_quality,
            style=style,
            move_breakdown=move_breakdown,
            recommendation=recommendation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "created_at": self.created_at,
            "player": self.player,
            "profile": self.profile.to_dict(),
            "data_quality": self.data_quality.to_dict(),
            "style": self.style.to_dict(),
            "move_breakdown": self.move_breakdown.to_dict(),
            "recommendation": self.recommendation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaModelCard":
        if data.get("schema_version") != MODEL_CARD_SCHEMA:
            raise ArtifactError(f"Unsupported model card schema: {data.get('schema_version')}")

        return cls(
            schema_version=data["schema_version"],
            package_version=data["package_version"],
            created_at=data["created_at"],
            player=data["player"],
            profile=PersonaProfile.from_dict(data["profile"]),
            data_quality=DataQualitySummary.from_dict(data["data_quality"]),
            style=StyleSummary.from_dict(data["style"]),
            move_breakdown=MoveBreakdown.from_dict(data["move_breakdown"]),
            recommendation=ModelCardRecommendation.from_dict(data["recommendation"]),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            output_path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(f"Unable to save model card: {output_path}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "PersonaModelCard":
        input_path = Path(path)
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load model card: {input_path}") from exc
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        tags = ", ".join(self.style.tags) if self.style.tags else "none"
        warnings = "\n".join(f"- {warning}" for warning in self.data_quality.warnings)
        notes = "\n".join(f"- {note}" for note in self.recommendation.notes)
        piece_rates = "\n".join(
            f"- {piece}: {_format_rate(rate)}"
            for piece, rate in sorted(self.move_breakdown.piece_rates.items())
        )

        return "\n".join(
            [
                f"# Persona Model Card: {self.player}",
                "",
                "## Data Quality",
                f"- Games: {self.data_quality.games}",
                f"- Examples: {self.data_quality.examples}",
                f"- Unique positions: {self.data_quality.unique_positions}",
                f"- Confidence: {self.data_quality.confidence}",
                "- Duplicate position rate: "
                f"{_format_rate(self.data_quality.duplicate_position_rate)}",
                f"- Average legal moves: {self.data_quality.average_legal_moves:.2f}",
                "",
                "## Style",
                f"- Tags: {tags}",
                f"- Forcing rate: {_format_rate(self.style.forcing_rate)}",
                f"- Capture rate: {_format_rate(self.style.capture_rate)}",
                f"- Check rate: {_format_rate(self.style.check_rate)}",
                f"- Castling rate: {_format_rate(self.style.castle_rate)}",
                f"- Opening phase rate: {_format_rate(self.style.opening_phase_rate)}",
                "",
                "## Piece Rates",
                piece_rates or "- none",
                "",
                "## Recommendation",
                f"- Model: {self.recommendation.recommended_model}",
                f"- Inference: {self.recommendation.recommended_inference}",
                f"- Neural readiness: {self.recommendation.neural_readiness}",
                "",
                "## Notes",
                notes or "- none",
                "",
                "## Warnings",
                warnings or "- none",
                "",
            ]
        )


def _format_rate(value: float) -> str:
    return f"{value:.1%}"
