from collections import Counter, defaultdict
from typing import Any

import chess

from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset.records import MoveExample
from persona_chess.models.scoring import legal_distribution_from_counter, predictions_from_scores
from persona_chess.models.types import MovePrediction

PhaseKey = str


class PhasePersonaModel:
    model_type = "phase"

    def __init__(self) -> None:
        self._phase_counts: dict[PhaseKey, Counter[str]] = defaultdict(Counter)
        self._global_counts: Counter[str] = Counter()

    def fit(self, examples: list[MoveExample]) -> None:
        self._phase_counts.clear()
        self._global_counts.clear()

        for example in examples:
            board = board_from_fen(example.fen)
            self._phase_counts[_phase_key(board)][example.move_uci] += 1
            self._global_counts[example.move_uci] += 1

    def predict(self, board: chess.Board, *, top_k: int = 1) -> list[MovePrediction]:
        scores = legal_distribution_from_counter(
            board, self._phase_counts.get(_phase_key(board), Counter())
        )
        reason = "phase_prior" if scores else "global_prior"

        if not scores:
            scores = legal_distribution_from_counter(board, self._global_counts)

        return [
            MovePrediction.from_board(board, move=move, score=score, reason=scored_reason)
            for move, score, scored_reason in predictions_from_scores(
                board,
                scores,
                top_k=top_k,
                reason=reason,
            )
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "phase_counts": {
                key: dict(counter) for key, counter in sorted(self._phase_counts.items())
            },
            "global_counts": dict(self._global_counts),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PhasePersonaModel":
        model = cls()
        model._phase_counts = defaultdict(
            Counter,
            {key: Counter(value) for key, value in payload.get("phase_counts", {}).items()},
        )
        model._global_counts = Counter(payload.get("global_counts", {}))
        return model


def _phase_key(board: chess.Board) -> PhaseKey:
    side = "w" if board.turn == chess.WHITE else "b"
    non_pawn_material = _non_pawn_material(board)

    if board.fullmove_number <= 12 and non_pawn_material >= 44:
        phase = "opening"
    elif non_pawn_material <= 18:
        phase = "endgame"
    else:
        phase = "middlegame"

    return f"{side}:{phase}"


def _non_pawn_material(board: chess.Board) -> int:
    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in values.items()
        for color in (chess.WHITE, chess.BLACK)
    )
