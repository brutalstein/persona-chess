from pathlib import Path

from persona_chess.dataset import SplitConfig, split_examples
from persona_chess.dataset.builder import build_move_examples
from persona_chess.pgn.filters import GameFilter

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_split_examples_keeps_games_together_by_default() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))

    split = split_examples(examples, SplitConfig(test_ratio=0.5, seed=7))

    train_games = {example.game_index for example in split.train}
    test_games = {example.game_index for example in split.test}

    assert train_games
    assert test_games
    assert train_games.isdisjoint(test_games)
    assert split.to_summary()["test_examples"] == len(split.test)
