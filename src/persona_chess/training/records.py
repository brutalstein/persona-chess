from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import chess

from persona_chess.chess.legal import board_from_fen, sorted_legal_moves
from persona_chess.dataset.records import MoveExample

TRAINING_RECORD_SCHEMA = "persona-chess/training-record/v1"

SideToMove = Literal["white", "black"]


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    schema_version: str
    fen: str
    position_key: str
    side_to_move: SideToMove
    legal_moves: tuple[str, ...]
    legal_move_count: int
    target_move: str
    target_index: int
    san: str
    player: str
    player_color: str
    game_index: int
    ply: int
    fullmove_number: int
    result: str

    @classmethod
    def from_example(cls, example: MoveExample) -> "TrainingRecord":
        board = board_from_fen(example.fen)
        legal_moves = tuple(move.uci() for move in sorted_legal_moves(board))

        try:
            target_index = legal_moves.index(example.move_uci)
        except ValueError as exc:
            raise ValueError(f"Target move is not legal in example: {example.move_uci}") from exc

        return cls(
            schema_version=TRAINING_RECORD_SCHEMA,
            fen=example.fen,
            position_key=example.position_key,
            side_to_move=_side_to_move(board),
            legal_moves=legal_moves,
            legal_move_count=len(legal_moves),
            target_move=example.move_uci,
            target_index=target_index,
            san=example.san,
            player=example.player,
            player_color=example.player_color,
            game_index=example.game_index,
            ply=example.ply,
            fullmove_number=example.fullmove_number,
            result=example.result,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["legal_moves"] = list(self.legal_moves)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingRecord":
        if data.get("schema_version") != TRAINING_RECORD_SCHEMA:
            raise ValueError(f"Unsupported training record schema: {data.get('schema_version')}")

        payload = dict(data)
        payload["legal_moves"] = tuple(payload["legal_moves"])
        return cls(**payload)


def build_training_records(examples: Iterable[MoveExample]) -> list[TrainingRecord]:
    return [TrainingRecord.from_example(example) for example in examples]


def _side_to_move(board: chess.Board) -> SideToMove:
    return "white" if board.turn == chess.WHITE else "black"
