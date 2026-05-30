from dataclasses import asdict, dataclass
from typing import Any

import chess


@dataclass(frozen=True, slots=True)
class MovePrediction:
    move_uci: str
    san: str
    score: float
    reason: str

    @classmethod
    def from_board(
        cls,
        board: chess.Board,
        *,
        move: chess.Move,
        score: float,
        reason: str,
    ) -> "MovePrediction":
        return cls(move_uci=move.uci(), san=board.san(move), score=score, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
