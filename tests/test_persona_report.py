from pathlib import Path

from persona_chess import PersonaChess
from persona_chess.dataset.builder import build_move_examples
from persona_chess.evaluation.persona_report import evaluate_persona_quality
from persona_chess.pgn.filters import GameFilter

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_persona_quality_report_compares_candidate_with_baseline() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    profile = PersonaChess().fit_pgn(FIXTURE, player="Target Player", model_type="blend").profile
    assert profile is not None
    candidate = PersonaChess().fit_examples(examples, profile=profile, model_type="blend")
    baseline = PersonaChess().fit_examples(examples[:5], profile=profile, model_type="frequency")

    report = evaluate_persona_quality(
        candidate.require_model(),
        examples,
        baseline_model=baseline.require_model(),
    )

    assert report.schema_version == "persona-chess/persona-evaluation-report/v1"
    assert report.move_matching.examples == len(examples)
    assert 0 <= report.style_similarity <= 1
    assert 0 <= report.opening_similarity <= 1
    assert report.comparison is not None
