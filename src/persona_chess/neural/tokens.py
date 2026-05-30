from dataclasses import dataclass

import chess

from persona_chess.chess.legal import board_from_fen
from persona_chess.training.records import TrainingRecord

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True, slots=True)
class PositionTokenizer:
    include_legal_moves: bool = False

    def tokenize_fen(self, fen: str) -> tuple[str, ...]:
        board = board_from_fen(fen)
        tokens = [
            BOS_TOKEN,
            f"turn:{'w' if board.turn == chess.WHITE else 'b'}",
            f"castling:{board.castling_xfen()}",
            f"ep:{_ep_square(board)}",
            f"phase:{_phase_bucket(board)}",
        ]

        tokens.extend(_piece_tokens(board))

        if self.include_legal_moves:
            tokens.extend(
                f"legal:{move.uci()}" for move in sorted(board.legal_moves, key=lambda m: m.uci())
            )

        tokens.append(EOS_TOKEN)
        return tuple(tokens)

    def tokenize_record(self, record: TrainingRecord) -> tuple[str, ...]:
        return self.tokenize_fen(record.fen)


def _piece_tokens(board: chess.Board) -> list[str]:
    tokens: list[str] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            tokens.append(f"{piece.symbol()}@{chess.square_name(square)}")
    return tokens


def _ep_square(board: chess.Board) -> str:
    return chess.square_name(board.ep_square) if board.ep_square is not None else "-"


def _phase_bucket(board: chess.Board) -> str:
    non_pawn_material = _non_pawn_material(board)
    if board.fullmove_number <= 12 and non_pawn_material >= 44:
        return "opening"
    if non_pawn_material <= 18:
        return "endgame"
    return "middlegame"


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
