import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from persona_chess.dataset.records import MoveExample


@dataclass(frozen=True, slots=True)
class SplitConfig:
    test_ratio: float = 0.2
    validation_ratio: float = 0.0
    seed: int = 42
    group_by_game: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("validation_ratio must be in [0, 1)")
        if not 0 < self.test_ratio < 1:
            raise ValueError("test_ratio must be in (0, 1)")
        if self.validation_ratio + self.test_ratio >= 1:
            raise ValueError("validation_ratio + test_ratio must be less than 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: list[MoveExample]
    validation: list[MoveExample]
    test: list[MoveExample]
    config: SplitConfig

    def to_summary(self) -> dict[str, Any]:
        return {
            "train_examples": len(self.train),
            "validation_examples": len(self.validation),
            "test_examples": len(self.test),
            "config": self.config.to_dict(),
        }


def split_examples(examples: list[MoveExample], config: SplitConfig | None = None) -> DatasetSplit:
    split_config = config or SplitConfig()
    if not examples:
        return DatasetSplit(train=[], validation=[], test=[], config=split_config)

    if split_config.group_by_game:
        return _split_grouped_by_game(examples, split_config)
    return _split_flat(examples, split_config)


def _split_grouped_by_game(examples: list[MoveExample], config: SplitConfig) -> DatasetSplit:
    grouped: dict[int, list[MoveExample]] = defaultdict(list)
    for example in examples:
        grouped[example.game_index].append(example)

    game_ids = sorted(grouped)
    rng = random.Random(config.seed)
    rng.shuffle(game_ids)

    validation_game_count = _ratio_count(len(game_ids), config.validation_ratio)
    test_game_count = _ratio_count(len(game_ids), config.test_ratio)

    validation_ids = set(game_ids[:validation_game_count])
    test_ids = set(game_ids[validation_game_count : validation_game_count + test_game_count])

    train: list[MoveExample] = []
    validation: list[MoveExample] = []
    test: list[MoveExample] = []

    for example in examples:
        if example.game_index in validation_ids:
            validation.append(example)
        elif example.game_index in test_ids:
            test.append(example)
        else:
            train.append(example)

    return DatasetSplit(train=train, validation=validation, test=test, config=config)


def _split_flat(examples: list[MoveExample], config: SplitConfig) -> DatasetSplit:
    shuffled = list(examples)
    rng = random.Random(config.seed)
    rng.shuffle(shuffled)

    validation_count = _ratio_count(len(shuffled), config.validation_ratio)
    test_count = _ratio_count(len(shuffled), config.test_ratio)

    validation = shuffled[:validation_count]
    test = shuffled[validation_count : validation_count + test_count]
    train = shuffled[validation_count + test_count :]
    return DatasetSplit(train=train, validation=validation, test=test, config=config)


def _ratio_count(total: int, ratio: float) -> int:
    if total <= 1 or ratio == 0:
        return 0
    return max(1, int(round(total * ratio)))
