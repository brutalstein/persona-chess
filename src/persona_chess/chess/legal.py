from collections.abc import Iterable

import chess


def board_from_fen(fen: str) -> chess.Board:
    if fen == "startpos":
        return chess.Board()
    return chess.Board(fen)


def legal_uci_moves(board: chess.Board) -> set[str]:
    return {move.uci() for move in board.legal_moves}


def position_key(board: chess.Board) -> str:
    turn = "w" if board.turn == chess.WHITE else "b"
    ep_square = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    return f"{board.board_fen()} {turn} {board.castling_xfen()} {ep_square}"


def sorted_legal_moves(board: chess.Board) -> list[chess.Move]:
    return sorted(board.legal_moves, key=lambda move: move.uci())


def legal_move_lookup(board: chess.Board, candidates: Iterable[str]) -> list[chess.Move]:
    legal = {move.uci(): move for move in board.legal_moves}
    return [legal[uci] for uci in candidates if uci in legal]
