from pathlib import Path

from persona_chess.dataset.builder import build_move_examples
from persona_chess.evaluation.metrics import evaluate_move_matching
from persona_chess.models import BlendPersonaModel
from persona_chess.pgn.filters import GameFilter

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_evaluation_metrics_include_rank_and_reason_distribution() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    model = BlendPersonaModel()
    model.fit(examples)

    metrics = evaluate_move_matching(model, examples, k=3)

    assert metrics.examples == len(examples)
    assert metrics.coverage == 1.0
    assert metrics.mean_rank >= 1.0
    assert metrics.mean_reciprocal_rank > 0
    assert metrics.reason_distribution
