from typing import Any, Protocol

import chess

from persona_chess.dataset.records import MoveExample
from persona_chess.models.types import MovePrediction


class PersonaModel(Protocol):
    model_type: str

    def fit(self, examples: list[MoveExample]) -> None: ...

    def predict(self, board: chess.Board, *, top_k: int = 1) -> list[MovePrediction]: ...

    def to_payload(self) -> dict[str, Any]: ...
