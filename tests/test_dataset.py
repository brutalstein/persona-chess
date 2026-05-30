from pathlib import Path

from persona_chess.dataset.builder import build_move_examples
from persona_chess.pgn.filters import GameFilter

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_build_move_examples_for_player() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))

    assert examples
    assert {example.player_color for example in examples} == {"white", "black"}
    assert examples[0].move_uci == "e2e4"
    assert examples[0].san == "e4"


def test_build_move_examples_can_filter_color() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player", color="black"))

    assert examples
    assert {example.player_color for example in examples} == {"black"}
    assert examples[0].move_uci == "g8f6"
