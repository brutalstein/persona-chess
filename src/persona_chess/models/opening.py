from collections import Counter, defaultdict
from typing import Any

import chess

from persona_chess.chess.legal import position_key
from persona_chess.dataset.records import MoveExample
from persona_chess.models.scoring import legal_distribution_from_counter, predictions_from_scores
from persona_chess.models.types import MovePrediction


class OpeningBookPersonaModel:
    model_type = "opening_book"

    def __init__(self, *, max_ply: int = 20) -> None:
        self.max_ply = max_ply
        self._book_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._global_counts: Counter[str] = Counter()

    def fit(self, examples: list[MoveExample]) -> None:
        self._book_counts.clear()
        self._global_counts.clear()

        for example in examples:
            self._global_counts[example.move_uci] += 1
            if example.ply <= self.max_ply:
                self._book_counts[example.position_key][example.move_uci] += 1

    def predict(self, board: chess.Board, *, top_k: int = 1) -> list[MovePrediction]:
        scores = legal_distribution_from_counter(
            board, self._book_counts.get(position_key(board), Counter())
        )
        reason = "opening_book" if scores else "global_prior"

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
            "max_ply": self.max_ply,
            "book_counts": {
                key: dict(counter) for key, counter in sorted(self._book_counts.items())
            },
            "global_counts": dict(self._global_counts),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OpeningBookPersonaModel":
        model = cls(max_ply=int(payload.get("max_ply", 20)))
        model._book_counts = defaultdict(
            Counter,
            {key: Counter(value) for key, value in payload.get("book_counts", {}).items()},
        )
        model._global_counts = Counter(payload.get("global_counts", {}))
        return model
