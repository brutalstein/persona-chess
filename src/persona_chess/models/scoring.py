from collections import Counter

import chess

from persona_chess.chess.legal import sorted_legal_moves


def predictions_from_scores(
    board: chess.Board,
    scores: dict[str, float],
    *,
    top_k: int,
    reason: str,
) -> list[tuple[chess.Move, float, str]]:
    legal_by_uci = {move.uci(): move for move in sorted_legal_moves(board)}
    scored = [
        (legal_by_uci[uci], score, reason)
        for uci, score in scores.items()
        if uci in legal_by_uci and score > 0
    ]

    if not scored and legal_by_uci:
        uniform_score = 1 / len(legal_by_uci)
        scored = [(move, uniform_score, "legal_fallback") for move in legal_by_uci.values()]

    scored.sort(key=lambda item: (-item[1], item[0].uci()))
    return scored[:top_k]


def legal_distribution_from_counter(board: chess.Board, counter: Counter[str]) -> dict[str, float]:
    legal_moves = sorted_legal_moves(board)
    total = sum(counter.get(move.uci(), 0) for move in legal_moves)
    if total == 0:
        return {}
    return {
        move.uci(): counter.get(move.uci(), 0) / total
        for move in legal_moves
        if counter.get(move.uci(), 0)
    }
