import pytest

from persona_chess.chess.legal import board_from_fen
from persona_chess.engines import (
    EngineCandidateScore,
    EngineGuidanceConfig,
    rerank_persona_predictions,
)
from persona_chess.models.types import MovePrediction


def test_engine_guided_reranking_can_prioritize_engine_quality() -> None:
    board = board_from_fen("startpos")
    predictions = [
        MovePrediction.from_board(
            board,
            move=board.parse_uci("e2e4"),
            score=0.90,
            reason="persona",
        ),
        MovePrediction.from_board(
            board,
            move=board.parse_uci("g1f3"),
            score=0.10,
            reason="persona",
        ),
    ]
    engine_scores = [
        EngineCandidateScore(move_uci="e2e4", centipawns=0, quality=0.05),
        EngineCandidateScore(move_uci="g1f3", centipawns=120, quality=0.95),
    ]

    reranked = rerank_persona_predictions(
        board,
        persona_predictions=predictions,
        engine_scores=engine_scores,
        top_k=2,
        engine_weight=0.80,
    )

    assert [prediction.move_uci for prediction in reranked] == ["g1f3", "e2e4"]
    assert reranked[0].reason == "engine_guided:persona"


def test_engine_guided_reranking_can_keep_persona_priority() -> None:
    board = board_from_fen("startpos")
    predictions = [
        MovePrediction.from_board(
            board,
            move=board.parse_uci("e2e4"),
            score=0.90,
            reason="persona",
        ),
        MovePrediction.from_board(
            board,
            move=board.parse_uci("g1f3"),
            score=0.10,
            reason="persona",
        ),
    ]
    engine_scores = [
        EngineCandidateScore(move_uci="e2e4", centipawns=0, quality=0.05),
        EngineCandidateScore(move_uci="g1f3", centipawns=120, quality=0.95),
    ]

    reranked = rerank_persona_predictions(
        board,
        persona_predictions=predictions,
        engine_scores=engine_scores,
        top_k=2,
        engine_weight=0.0,
    )

    assert [prediction.move_uci for prediction in reranked] == ["e2e4", "g1f3"]


def test_engine_guidance_config_validates_values() -> None:
    with pytest.raises(ValueError):
        EngineGuidanceConfig(engine_weight=1.1)

    with pytest.raises(ValueError):
        EngineGuidanceConfig(candidate_count=0)

    with pytest.raises(ValueError):
        EngineGuidanceConfig(time_limit=0)
