from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset.records import MoveExample
from persona_chess.models.base import PersonaModel


@dataclass(frozen=True, slots=True)
class MoveMatchMetrics:
    examples: int
    top_1: float
    top_k: float
    k: int
    coverage: float
    mean_rank: float
    mean_reciprocal_rank: float
    reason_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveMatchMetrics":
        payload = dict(data)
        payload["reason_distribution"] = dict(payload["reason_distribution"])
        return cls(**payload)


def evaluate_move_matching(
    model: PersonaModel,
    examples: list[MoveExample],
    *,
    k: int = 3,
) -> MoveMatchMetrics:
    if not examples:
        return MoveMatchMetrics(
            examples=0,
            top_1=0.0,
            top_k=0.0,
            k=k,
            coverage=0.0,
            mean_rank=0.0,
            mean_reciprocal_rank=0.0,
            reason_distribution={},
        )

    top_1_matches = 0
    top_k_matches = 0
    covered = 0
    rank_sum = 0.0
    reciprocal_rank_sum = 0.0
    reasons: Counter[str] = Counter()

    for example in examples:
        predictions = model.predict(board_from_fen(example.fen), top_k=k)
        predicted_moves = [prediction.move_uci for prediction in predictions]
        covered += int(bool(predictions))
        reasons[predictions[0].reason if predictions else "no_prediction"] += 1
        top_1_matches += int(bool(predicted_moves) and predicted_moves[0] == example.move_uci)
        top_k_matches += int(example.move_uci in predicted_moves)

        if example.move_uci in predicted_moves:
            rank = predicted_moves.index(example.move_uci) + 1
            reciprocal_rank_sum += 1 / rank
        else:
            rank = k + 1
        rank_sum += rank

    total = len(examples)
    return MoveMatchMetrics(
        examples=total,
        top_1=top_1_matches / total,
        top_k=top_k_matches / total,
        k=k,
        coverage=covered / total,
        mean_rank=rank_sum / total,
        mean_reciprocal_rank=reciprocal_rank_sum / total,
        reason_distribution=dict(reasons.most_common()),
    )
