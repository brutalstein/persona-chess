from pathlib import Path

from persona_chess.pgn.filters import GameFilter
from persona_chess.profile.builder import build_profile

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_build_profile() -> None:
    profile = build_profile(FIXTURE, GameFilter(player="Target Player"))

    assert profile.player == "Target Player"
    assert profile.games == 2
    assert profile.white_games == 1
    assert profile.black_games == 1
    assert profile.result_distribution == {"win": 2}
    assert profile.first_move_distribution["e4"] == 1
    assert profile.first_move_distribution["Nf6"] == 1
