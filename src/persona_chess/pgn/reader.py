import bz2
import gzip
import lzma
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from io import TextIOWrapper
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn

from persona_chess.exceptions import OptionalDependencyError, PgnReadError, PlayerNotFoundError
from persona_chess.pgn.filters import GameFilter, PlayerColor


@dataclass(frozen=True, slots=True)
class PgnGame:
    headers: dict[str, str]
    moves: tuple[chess.Move, ...]

    @property
    def white(self) -> str:
        return self.headers.get("White", "")

    @property
    def black(self) -> str:
        return self.headers.get("Black", "")

    @property
    def result(self) -> str:
        return self.headers.get("Result", "*")

    @property
    def variant(self) -> str:
        return self.headers.get("Variant", "Standard")


@dataclass(frozen=True, slots=True)
class PlayerGame:
    game: PgnGame
    player: str
    color: PlayerColor
    index: int


def iter_pgn_games(path: str | Path) -> Iterator[PgnGame]:
    pgn_path = Path(path)
    try:
        with open_chess_text(pgn_path) as handle:
            while game := chess.pgn.read_game(handle):
                yield PgnGame(
                    headers={str(key): str(value) for key, value in game.headers.items()},
                    moves=tuple(game.mainline_moves()),
                )
    except OSError as exc:
        raise PgnReadError(f"Unable to read PGN file: {pgn_path}") from exc


@contextmanager
def open_chess_text(path: str | Path) -> Iterator[TextIO]:
    input_path = Path(path)
    suffixes = tuple(suffix.casefold() for suffix in input_path.suffixes)
    if suffixes[-1:] == (".gz",):
        with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as handle:
            yield handle
        return
    if suffixes[-1:] == (".bz2",):
        with bz2.open(input_path, "rt", encoding="utf-8", errors="replace") as handle:
            yield handle
        return
    if suffixes[-1:] in {(".xz",), (".lzma",)}:
        with lzma.open(input_path, "rt", encoding="utf-8", errors="replace") as handle:
            yield handle
        return
    if suffixes[-1:] == (".zst",):
        with _open_zstandard_text(input_path) as handle:
            yield handle
        return
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        yield handle


@contextmanager
def _open_zstandard_text(path: Path) -> Iterator[TextIO]:
    try:
        zstandard = import_module("zstandard")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "Reading .zst PGN files requires zstandard. Install persona-chess with: "
            "pip install persona-chess"
        ) from exc

    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = TextIOWrapper(reader, encoding="utf-8", errors="replace")
        try:
            yield text
        finally:
            text.close()


def iter_player_games(path: str | Path, game_filter: GameFilter) -> Iterator[PlayerGame]:
    matched = 0
    target = game_filter.normalized_player()

    for index, game in enumerate(iter_pgn_games(path), start=1):
        if not game_filter.include_variants and game.variant.casefold() != "standard":
            continue

        color = _matching_color(game, target)
        if color is None:
            continue

        if game_filter.color != "both" and color != game_filter.color:
            continue

        matched += 1
        yield PlayerGame(game=game, player=game_filter.player, color=color, index=index)

        if game_filter.max_games is not None and matched >= game_filter.max_games:
            break

    if matched == 0:
        raise PlayerNotFoundError(f"No games found for player: {game_filter.player}")


def _matching_color(game: PgnGame, normalized_player: str) -> PlayerColor | None:
    if game.white.casefold().strip() == normalized_player:
        return "white"
    if game.black.casefold().strip() == normalized_player:
        return "black"
    return None
