import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

from persona_chess.models.base import PersonaModel
from persona_chess.models.types import MovePrediction


@dataclass(frozen=True, slots=True)
class EngineGuidanceConfig:
    engine_weight: float = 0.35
    candidate_count: int = 12
    time_limit: float | None = 0.05
    depth: int | None = None
    mate_score: int = 100_000
    score_temperature: float = 250.0

    def __post_init__(self) -> None:
        if not 0 <= self.engine_weight <= 1:
            raise ValueError("engine_weight must be in [0, 1]")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be positive when set")
        if self.depth is not None and self.depth <= 0:
            raise ValueError("depth must be positive when set")
        if self.mate_score <= 0:
            raise ValueError("mate_score must be positive")
        if self.score_temperature <= 0:
            raise ValueError("score_temperature must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EngineCandidateScore:
    move_uci: str
    centipawns: int
    quality: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_engine_guided_moves(
    model: PersonaModel,
    *,
    board: chess.Board,
    engine_path: str | Path,
    top_k: int = 3,
    config: EngineGuidanceConfig | None = None,
) -> list[MovePrediction]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    active_config = config or EngineGuidanceConfig()
    candidate_count = max(top_k, active_config.candidate_count)
    persona_predictions = model.predict(board, top_k=candidate_count)
    if not persona_predictions:
        return []

    engine_scores = score_persona_candidates_with_uci_engine(
        board,
        moves=[chess.Move.from_uci(prediction.move_uci) for prediction in persona_predictions],
        engine_path=engine_path,
        config=active_config,
    )
    return rerank_persona_predictions(
        board,
        persona_predictions=persona_predictions,
        engine_scores=engine_scores,
        top_k=top_k,
        engine_weight=active_config.engine_weight,
    )


def score_persona_candidates_with_uci_engine(
    board: chess.Board,
    *,
    moves: list[chess.Move],
    engine_path: str | Path,
    config: EngineGuidanceConfig | None = None,
) -> list[EngineCandidateScore]:
    if not moves:
        return []

    active_config = config or EngineGuidanceConfig()
    raw_scores: list[tuple[chess.Move, int]] = []
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    try:
        for move in moves:
            if move not in board.legal_moves:
                continue
            raw_scores.append(
                (
                    move,
                    _score_candidate_move(
                        engine,
                        board=board,
                        move=move,
                        config=active_config,
                    ),
                )
            )
    finally:
        engine.quit()

    qualities = _softmax_scores(
        [score for _, score in raw_scores],
        temperature=active_config.score_temperature,
    )
    return [
        EngineCandidateScore(move_uci=move.uci(), centipawns=score, quality=quality)
        for (move, score), quality in zip(raw_scores, qualities, strict=True)
    ]


def rerank_persona_predictions(
    board: chess.Board,
    *,
    persona_predictions: list[MovePrediction],
    engine_scores: list[EngineCandidateScore],
    top_k: int = 3,
    engine_weight: float = 0.35,
) -> list[MovePrediction]:
    if not 0 <= engine_weight <= 1:
        raise ValueError("engine_weight must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    engine_quality = {score.move_uci: score.quality for score in engine_scores}
    persona_quality = _normalize_prediction_scores(persona_predictions)

    reranked: list[MovePrediction] = []
    for prediction in persona_predictions:
        score = (1 - engine_weight) * persona_quality.get(
            prediction.move_uci, 0.0
        ) + engine_weight * engine_quality.get(prediction.move_uci, 0.0)
        move = chess.Move.from_uci(prediction.move_uci)
        if move not in board.legal_moves:
            continue
        reranked.append(
            MovePrediction.from_board(
                board,
                move=move,
                score=score,
                reason=f"engine_guided:{prediction.reason}",
            )
        )

    reranked.sort(key=lambda prediction: (-prediction.score, prediction.move_uci))
    return reranked[:top_k]


def _score_candidate_move(
    engine: chess.engine.SimpleEngine,
    *,
    board: chess.Board,
    move: chess.Move,
    config: EngineGuidanceConfig,
) -> int:
    child = board.copy(stack=False)
    child.push(move)
    info = engine.analyse(child, _engine_limit(config))
    score = info["score"].pov(board.turn)
    return int(score.score(mate_score=config.mate_score) or 0)


def _engine_limit(config: EngineGuidanceConfig) -> chess.engine.Limit:
    return chess.engine.Limit(time=config.time_limit, depth=config.depth)


def _normalize_prediction_scores(predictions: list[MovePrediction]) -> dict[str, float]:
    if not predictions:
        return {}

    total = sum(max(0.0, prediction.score) for prediction in predictions)
    if total > 0:
        return {
            prediction.move_uci: max(0.0, prediction.score) / total for prediction in predictions
        }

    uniform_score = 1 / len(predictions)
    return {prediction.move_uci: uniform_score for prediction in predictions}


def _softmax_scores(scores: list[int], *, temperature: float) -> list[float]:
    if not scores:
        return []

    scaled = [score / temperature for score in scores]
    max_scaled = max(scaled)
    exp_scores = [math.exp(score - max_scaled) for score in scaled]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]
