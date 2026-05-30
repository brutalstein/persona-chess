from dataclasses import asdict, dataclass
from typing import Any

import chess

from persona_chess.chess.legal import position_key
from persona_chess.pgn.filters import PlayerColor

MOVE_EXAMPLE_SCHEMA = "persona-chess/move-example/v1"


@dataclass(frozen=True, slots=True)
class MoveExample:
    schema_version: str
    fen: str
    position_key: str
    move_uci: str
    san: str
    player: str
    player_color: PlayerColor
    game_index: int
    ply: int
    fullmove_number: int
    result: str
    white: str
    black: str

    @classmethod
    def from_board(
        cls,
        *,
        board: chess.Board,
        move: chess.Move,
        player: str,
        player_color: PlayerColor,
        game_index: int,
        ply: int,
        result: str,
        white: str,
        black: str,
    ) -> "MoveExample":
        return cls(
            schema_version=MOVE_EXAMPLE_SCHEMA,
            fen=board.fen(),
            position_key=position_key(board),
            move_uci=move.uci(),
            san=board.san(move),
            player=player,
            player_color=player_color,
            game_index=game_index,
            ply=ply,
            fullmove_number=board.fullmove_number,
            result=result,
            white=white,
            black=black,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveExample":
        payload = dict(data)
        payload.setdefault("schema_version", MOVE_EXAMPLE_SCHEMA)
        if payload["schema_version"] != MOVE_EXAMPLE_SCHEMA:
            raise ValueError(f"Unsupported move example schema: {payload['schema_version']}")
        return cls(**payload)
