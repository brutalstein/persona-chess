from typing import Any

import chess

from persona_chess.chess.legal import sorted_legal_moves
from persona_chess.dataset.records import MoveExample
from persona_chess.models.frequency import FrequencyPersonaModel
from persona_chess.models.opening import OpeningBookPersonaModel
from persona_chess.models.phase import PhasePersonaModel
from persona_chess.models.types import MovePrediction


class BlendPersonaModel:
    model_type = "blend"

    def __init__(
        self,
        *,
        frequency_weight: float = 0.55,
        opening_weight: float = 0.25,
        phase_weight: float = 0.20,
    ) -> None:
        self.frequency_weight = frequency_weight
        self.opening_weight = opening_weight
        self.phase_weight = phase_weight
        self.frequency = FrequencyPersonaModel()
        self.opening = OpeningBookPersonaModel()
        self.phase = PhasePersonaModel()

    def fit(self, examples: list[MoveExample]) -> None:
        self.frequency.fit(examples)
        self.opening.fit(examples)
        self.phase.fit(examples)

    def predict(self, board: chess.Board, *, top_k: int = 1) -> list[MovePrediction]:
        legal_count = len(list(board.legal_moves))
        if legal_count == 0:
            return []

        scores: dict[str, float] = {move.uci(): 0.0 for move in sorted_legal_moves(board)}
        reasons: dict[str, list[str]] = {move_uci: [] for move_uci in scores}

        for weight, predictions in (
            (self.frequency_weight, self.frequency.predict(board, top_k=legal_count)),
            (self.opening_weight, self.opening.predict(board, top_k=legal_count)),
            (self.phase_weight, self.phase.predict(board, top_k=legal_count)),
        ):
            for prediction in predictions:
                scores[prediction.move_uci] += weight * prediction.score
                reasons[prediction.move_uci].append(prediction.reason)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            MovePrediction.from_board(
                board,
                move=chess.Move.from_uci(move_uci),
                score=score,
                reason=_blend_reason(reasons[move_uci]),
            )
            for move_uci, score in ranked[:top_k]
            if score > 0
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "weights": {
                "frequency": self.frequency_weight,
                "opening": self.opening_weight,
                "phase": self.phase_weight,
            },
            "models": {
                "frequency": self.frequency.to_payload(),
                "opening": self.opening.to_payload(),
                "phase": self.phase.to_payload(),
            },
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BlendPersonaModel":
        weights = payload.get("weights", {})
        model = cls(
            frequency_weight=float(weights.get("frequency", 0.55)),
            opening_weight=float(weights.get("opening", 0.25)),
            phase_weight=float(weights.get("phase", 0.20)),
        )
        models = payload.get("models", {})
        model.frequency = FrequencyPersonaModel.from_payload(models.get("frequency", {}))
        model.opening = OpeningBookPersonaModel.from_payload(models.get("opening", {}))
        model.phase = PhasePersonaModel.from_payload(models.get("phase", {}))
        return model


def _blend_reason(reasons: list[str]) -> str:
    unique_reasons = sorted(set(reasons))
    if not unique_reasons:
        return "blend"
    return "blend:" + "+".join(unique_reasons)
