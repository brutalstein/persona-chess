import gzip
from pathlib import Path

from persona_chess.dataset.builder import build_move_examples
from persona_chess.pgn.filters import GameFilter
from persona_chess.pgn.reader import iter_pgn_games

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_reader_accepts_gzip_pgn(tmp_path: Path) -> None:
    compressed = tmp_path / "sample.pgn.gz"
    compressed.write_bytes(gzip.compress(FIXTURE.read_bytes()))

    games = list(iter_pgn_games(compressed))
    examples = build_move_examples(compressed, GameFilter(player="Target Player"))

    assert len(games) == 2
    assert len(examples) == 10
