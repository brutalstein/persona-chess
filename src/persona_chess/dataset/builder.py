from collections.abc import Iterator
from pathlib import Path

import chess

from persona_chess.dataset.records import MoveExample
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.pgn.reader import PgnGame, iter_pgn_games, iter_player_games


def build_move_examples(
    path: str | Path,
    game_filter: GameFilter,
    *,
    skip_first_plies: int = 0,
    max_examples: int | None = None,
) -> list[MoveExample]:
    return list(
        iter_move_examples(
            path,
            game_filter,
            skip_first_plies=skip_first_plies,
            max_examples=max_examples,
        )
    )


def iter_move_examples(
    path: str | Path,
    game_filter: GameFilter,
    *,
    skip_first_plies: int = 0,
    max_examples: int | None = None,
) -> Iterator[MoveExample]:
    emitted = 0
    for player_game in iter_player_games(path, game_filter):
        board = chess.Board()
        target_turn = chess.WHITE if player_game.color == "white" else chess.BLACK

        for ply, move in enumerate(player_game.game.moves, start=1):
            if board.turn == target_turn and ply > skip_first_plies:
                yield MoveExample.from_board(
                    board=board,
                    move=move,
                    player=player_game.player,
                    player_color=player_game.color,
                    game_index=player_game.index,
                    ply=ply,
                    result=player_game.game.result,
                    white=player_game.game.white,
                    black=player_game.game.black,
                )
                emitted += 1

                if max_examples is not None and emitted >= max_examples:
                    return

            board.push(move)


def iter_all_move_examples(
    path: str | Path,
    *,
    skip_first_plies: int = 0,
    max_games: int | None = None,
    max_examples: int | None = None,
    include_variants: bool = False,
) -> Iterator[MoveExample]:
    emitted = 0
    matched_games = 0
    for game_index, game in enumerate(iter_pgn_games(path), start=1):
        if not include_variants and game.variant.casefold() != "standard":
            continue
        matched_games += 1
        for example in _iter_game_move_examples(
            game,
            game_index=game_index,
            skip_first_plies=skip_first_plies,
        ):
            yield example
            emitted += 1
            if max_examples is not None and emitted >= max_examples:
                return

        if max_games is not None and matched_games >= max_games:
            return


def _iter_game_move_examples(
    game: PgnGame,
    *,
    game_index: int,
    skip_first_plies: int,
) -> Iterator[MoveExample]:
    board = chess.Board()
    for ply, move in enumerate(game.moves, start=1):
        if ply > skip_first_plies:
            player_color: PlayerColor = "white" if board.turn == chess.WHITE else "black"
            yield MoveExample.from_board(
                board=board,
                move=move,
                player=game.white if player_color == "white" else game.black,
                player_color=player_color,
                game_index=game_index,
                ply=ply,
                result=game.result,
                white=game.white,
                black=game.black,
            )
        board.push(move)
