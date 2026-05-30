from collections import Counter, defaultdict
from typing import Any

import chess

from persona_chess.chess.legal import position_key, sorted_legal_moves
from persona_chess.dataset.records import MoveExample
from persona_chess.models.types import MovePrediction


class FrequencyPersonaModel:
    model_type = "frequency"

    def __init__(self) -> None:
        self._position_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._global_counts: Counter[str] = Counter()

    def fit(self, examples: list[MoveExample]) -> None:
        self._position_counts.clear()
        self._global_counts.clear()

        for example in examples:
            self._position_counts[example.position_key][example.move_uci] += 1
            self._global_counts[example.move_uci] += 1

    def predict(self, board: chess.Board, *, top_k: int = 1) -> list[MovePrediction]:
        legal_moves = sorted_legal_moves(board)
        if not legal_moves:
            return []

        key = position_key(board)
        source_counts = self._position_counts.get(key)
        reason = "position_memory" if source_counts else "global_prior"

        scored: list[tuple[chess.Move, float]] = []
        if source_counts:
            total = sum(source_counts.values())
            for move in legal_moves:
                count = source_counts.get(move.uci(), 0)
                if count:
                    scored.append((move, count / total))

        if not scored:
            legal_total = sum(self._global_counts.get(move.uci(), 0) for move in legal_moves)
            if legal_total:
                scored = [
                    (move, self._global_counts.get(move.uci(), 0) / legal_total)
                    for move in legal_moves
                    if self._global_counts.get(move.uci(), 0)
                ]

        if not scored:
            reason = "legal_fallback"
            uniform_score = 1 / len(legal_moves)
            scored = [(move, uniform_score) for move in legal_moves]

        scored.sort(key=lambda item: (-item[1], item[0].uci()))
        return [
            MovePrediction.from_board(board, move=move, score=score, reason=reason)
            for move, score in scored[:top_k]
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "position_counts": {
                key: dict(counter) for key, counter in sorted(self._position_counts.items())
            },
            "global_counts": dict(self._global_counts),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FrequencyPersonaModel":
        model = cls()
        model._position_counts = defaultdict(
            Counter,
            {key: Counter(value) for key, value in payload.get("position_counts", {}).items()},
        )
        model._global_counts = Counter(payload.get("global_counts", {}))
        return model
