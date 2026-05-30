from collections import Counter
from pathlib import Path

import chess

from persona_chess.pgn.filters import GameFilter
from persona_chess.pgn.reader import PlayerGame, iter_player_games
from persona_chess.profile.types import MoveTendencies, PersonaProfile


def build_profile(path: str | Path, game_filter: GameFilter) -> PersonaProfile:
    games = list(iter_player_games(path, game_filter))
    target_moves = 0
    captures = 0
    checks = 0
    kingside_castles = 0
    queenside_castles = 0
    promotions = 0
    opening_sequences: Counter[str] = Counter()
    first_moves: Counter[str] = Counter()
    results: Counter[str] = Counter()

    for player_game in games:
        board = chess.Board()
        target_turn = chess.WHITE if player_game.color == "white" else chess.BLACK
        game_opening: list[str] = []
        first_target_move_seen = False

        for ply, move in enumerate(player_game.game.moves, start=1):
            san = board.san(move)
            if ply <= 8:
                game_opening.append(san)

            if board.turn == target_turn:
                target_moves += 1
                captures += int(board.is_capture(move))
                promotions += int(move.promotion is not None)

                if board.is_castling(move):
                    if chess.square_file(move.to_square) > chess.square_file(move.from_square):
                        kingside_castles += 1
                    else:
                        queenside_castles += 1

                if not first_target_move_seen:
                    first_moves[san] += 1
                    first_target_move_seen = True

                board.push(move)
                checks += int(board.is_check())
                continue

            board.push(move)

        if game_opening:
            opening_sequences[" ".join(game_opening)] += 1
        results[_result_for_player(player_game)] += 1

    tendencies = MoveTendencies.from_counts(
        moves=target_moves,
        captures=captures,
        checks=checks,
        kingside_castles=kingside_castles,
        queenside_castles=queenside_castles,
        promotions=promotions,
    )

    return PersonaProfile(
        player=game_filter.player,
        games=len(games),
        white_games=sum(1 for game in games if game.color == "white"),
        black_games=sum(1 for game in games if game.color == "black"),
        target_moves=target_moves,
        tendencies=tendencies,
        result_distribution=dict(results.most_common()),
        first_move_distribution=dict(first_moves.most_common(10)),
        opening_distribution=dict(opening_sequences.most_common(10)),
    )


def _result_for_player(player_game: PlayerGame) -> str:
    result = player_game.game.result
    if result == "1/2-1/2":
        return "draw"
    if result == "*":
        return "unknown"
    if (result == "1-0" and player_game.color == "white") or (
        result == "0-1" and player_game.color == "black"
    ):
        return "win"
    return "loss"
