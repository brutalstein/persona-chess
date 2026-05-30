from pathlib import Path

from persona_chess.dataset import SplitConfig, split_examples
from persona_chess.dataset.builder import build_move_examples
from persona_chess.evaluation.metrics import evaluate_move_matching
from persona_chess.evaluation.report import BenchmarkReport, SplitEvaluation
from persona_chess.facade import PersonaChess
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.profile.builder import build_profile


def run_benchmark(
    pgn: str | Path,
    *,
    player: str,
    model_type: str = "blend",
    color: PlayerColor = "both",
    test_ratio: float = 0.2,
    validation_ratio: float = 0.0,
    seed: int = 42,
    k: int = 3,
) -> BenchmarkReport:
    game_filter = GameFilter(player=player, color=color)
    examples = build_move_examples(pgn, game_filter)
    profile = build_profile(pgn, game_filter)
    dataset_split = split_examples(
        examples,
        SplitConfig(test_ratio=test_ratio, validation_ratio=validation_ratio, seed=seed),
    )

    persona = PersonaChess().fit_examples(
        dataset_split.train,
        profile=profile,
        model_type=model_type,
    )
    model = persona.require_model()
    evaluations = [
        SplitEvaluation(
            name="train",
            metrics=evaluate_move_matching(model, dataset_split.train, k=k),
        )
    ]

    if dataset_split.validation:
        evaluations.append(
            SplitEvaluation(
                name="validation",
                metrics=evaluate_move_matching(model, dataset_split.validation, k=k),
            )
        )

    evaluations.append(
        SplitEvaluation(
            name="test",
            metrics=evaluate_move_matching(model, dataset_split.test, k=k),
        )
    )

    return BenchmarkReport.create(
        model=model,
        profile=profile,
        dataset_split=dataset_split,
        evaluations=evaluations,
    )
