import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from persona_chess._version import __version__
from persona_chess.dataset.split import DatasetSplit
from persona_chess.evaluation.metrics import MoveMatchMetrics
from persona_chess.exceptions import ArtifactError
from persona_chess.models.base import PersonaModel
from persona_chess.profile.types import PersonaProfile

BENCHMARK_REPORT_SCHEMA = "persona-chess/benchmark-report/v1"


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    name: str
    metrics: MoveMatchMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitEvaluation":
        return cls(
            name=str(data["name"]),
            metrics=MoveMatchMetrics.from_dict(data["metrics"]),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    schema_version: str
    package_version: str
    created_at: str
    model_type: str
    player: str
    profile: PersonaProfile
    split: dict[str, Any]
    evaluations: tuple[SplitEvaluation, ...]

    @classmethod
    def create(
        cls,
        *,
        model: PersonaModel,
        profile: PersonaProfile,
        dataset_split: DatasetSplit,
        evaluations: list[SplitEvaluation],
    ) -> "BenchmarkReport":
        return cls(
            schema_version=BENCHMARK_REPORT_SCHEMA,
            package_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_type=model.model_type,
            player=profile.player,
            profile=profile,
            split=dataset_split.to_summary(),
            evaluations=tuple(evaluations),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["profile"] = self.profile.to_dict()
        data["evaluations"] = [evaluation.to_dict() for evaluation in self.evaluations]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReport":
        if data.get("schema_version") != BENCHMARK_REPORT_SCHEMA:
            raise ArtifactError(
                f"Unsupported benchmark report schema: {data.get('schema_version')}"
            )

        return cls(
            schema_version=data["schema_version"],
            package_version=data["package_version"],
            created_at=data["created_at"],
            model_type=data["model_type"],
            player=data["player"],
            profile=PersonaProfile.from_dict(data["profile"]),
            split=data["split"],
            evaluations=tuple(
                SplitEvaluation.from_dict(evaluation) for evaluation in data["evaluations"]
            ),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            output_path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(f"Unable to save benchmark report: {output_path}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkReport":
        input_path = Path(path)
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load benchmark report: {input_path}") from exc
        return cls.from_dict(data)
