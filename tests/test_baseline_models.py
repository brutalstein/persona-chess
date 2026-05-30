from pathlib import Path

import chess

from persona_chess.dataset.builder import build_move_examples
from persona_chess.models import BlendPersonaModel, OpeningBookPersonaModel, PhasePersonaModel
from persona_chess.pgn.filters import GameFilter

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_opening_book_model_predicts_memorized_opening_move() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    model = OpeningBookPersonaModel()
    model.fit(examples)

    prediction = model.predict(chess.Board(), top_k=1)[0].move_uci

    assert prediction == "e2e4"


def test_phase_model_returns_legal_prediction() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    model = PhasePersonaModel()
    model.fit(examples)

    prediction = model.predict(chess.Board(), top_k=1)[0].move_uci

    assert prediction in {"e2e4", "g1f3", "d2d4"}


def test_blend_model_round_trips_payload() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    model = BlendPersonaModel()
    model.fit(examples)

    loaded = BlendPersonaModel.from_payload(model.to_payload())

    assert loaded.predict(chess.Board(), top_k=1) == model.predict(chess.Board(), top_k=1)
