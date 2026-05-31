import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import chess

from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset.records import MoveExample
from persona_chess.engines import EngineGuidanceConfig
from persona_chess.evaluation.metrics import MoveMatchMetrics, evaluate_move_matching
from persona_chess.models.base import PersonaModel

PERSONA_EVALUATION_REPORT_SCHEMA = "persona-chess/persona-evaluation-report/v1"


@dataclass(frozen=True, slots=True)
class StyleVector:
    capture_rate: float
    check_rate: float
    castle_rate: float
    promotion_rate: float
    forcing_rate: float
    early_queen_rate: float
    opening_phase_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelComparisonMetrics:
    agreement_top1: float
    candidate_top1_delta: float
    candidate_topk_delta: float
    baseline_top_1: float | None
    baseline_top_k: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EngineQualityMetrics:
    examples: int
    average_model_centipawns: float
    average_target_centipawns: float
    average_centipawn_loss: float
    blunder_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DistributionSimilarityMetrics:
    histogram_overlap: float
    cosine_similarity: float
    jensen_shannon_distance: float | None
    wasserstein_distance: float | None
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionConfidenceMetrics:
    examples: int
    average_top1_score: float
    average_correct_top1_score: float
    average_incorrect_top1_score: float
    average_top1_margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SegmentEvaluation:
    name: str
    segment_type: str
    examples: int
    metrics: MoveMatchMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "segment_type": self.segment_type,
            "examples": self.examples,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PersonaEvaluationReport:
    schema_version: str
    candidate_model_type: str
    examples: int
    move_matching: MoveMatchMetrics
    actual_style: StyleVector
    predicted_style: StyleVector
    style_similarity: float
    opening_similarity: float
    phase_metrics: tuple[SegmentEvaluation, ...]
    piece_metrics: tuple[SegmentEvaluation, ...]
    style_distribution: DistributionSimilarityMetrics
    opening_distribution: DistributionSimilarityMetrics
    confidence: PredictionConfidenceMetrics
    comparison: ModelComparisonMetrics | None = None
    engine_quality: EngineQualityMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["move_matching"] = self.move_matching.to_dict()
        data["actual_style"] = self.actual_style.to_dict()
        data["predicted_style"] = self.predicted_style.to_dict()
        data["phase_metrics"] = [metric.to_dict() for metric in self.phase_metrics]
        data["piece_metrics"] = [metric.to_dict() for metric in self.piece_metrics]
        data["style_distribution"] = self.style_distribution.to_dict()
        data["opening_distribution"] = self.opening_distribution.to_dict()
        data["confidence"] = self.confidence.to_dict()
        data["comparison"] = self.comparison.to_dict() if self.comparison is not None else None
        data["engine_quality"] = (
            self.engine_quality.to_dict() if self.engine_quality is not None else None
        )
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def evaluate_persona_quality(
    model: PersonaModel,
    examples: list[MoveExample],
    *,
    baseline_model: PersonaModel | None = None,
    k: int = 3,
    engine_path: str | Path | None = None,
    engine_limit: EngineGuidanceConfig | None = None,
    blunder_threshold_cp: int = 200,
) -> PersonaEvaluationReport:
    move_matching = evaluate_move_matching(model, examples, k=k)
    actual_style = style_vector_from_examples(examples)
    predicted_examples = _prediction_examples(model, examples)
    predicted_style = style_vector_from_examples(predicted_examples)
    actual_openings = _opening_counter(examples)
    predicted_openings = _opening_counter(predicted_examples)
    comparison = (
        compare_models(model, baseline_model, examples, candidate_metrics=move_matching, k=k)
        if baseline_model is not None
        else None
    )
    engine_quality = (
        evaluate_engine_quality(
            model,
            examples,
            engine_path=engine_path,
            config=engine_limit or EngineGuidanceConfig(),
            blunder_threshold_cp=blunder_threshold_cp,
        )
        if engine_path is not None
        else None
    )
    return PersonaEvaluationReport(
        schema_version=PERSONA_EVALUATION_REPORT_SCHEMA,
        candidate_model_type=model.model_type,
        examples=len(examples),
        move_matching=move_matching,
        actual_style=actual_style,
        predicted_style=predicted_style,
        style_similarity=style_similarity(actual_style, predicted_style),
        opening_similarity=_histogram_overlap(actual_openings, predicted_openings),
        phase_metrics=tuple(_segment_metrics(model, examples, k=k, segment_type="phase")),
        piece_metrics=tuple(_segment_metrics(model, examples, k=k, segment_type="piece")),
        style_distribution=distribution_similarity(
            _style_counter(actual_style),
            _style_counter(predicted_style),
        ),
        opening_distribution=distribution_similarity(actual_openings, predicted_openings),
        confidence=prediction_confidence(model, examples),
        comparison=comparison,
        engine_quality=engine_quality,
    )


def compare_models(
    candidate_model: PersonaModel,
    baseline_model: PersonaModel,
    examples: list[MoveExample],
    *,
    candidate_metrics: MoveMatchMetrics | None = None,
    k: int = 3,
) -> ModelComparisonMetrics:
    candidate = candidate_metrics or evaluate_move_matching(candidate_model, examples, k=k)
    baseline = evaluate_move_matching(baseline_model, examples, k=k)
    agreement = _top1_agreement(candidate_model, baseline_model, examples)
    return ModelComparisonMetrics(
        agreement_top1=agreement,
        candidate_top1_delta=candidate.top_1 - baseline.top_1,
        candidate_topk_delta=candidate.top_k - baseline.top_k,
        baseline_top_1=baseline.top_1,
        baseline_top_k=baseline.top_k,
    )


def style_vector_from_examples(examples: list[MoveExample]) -> StyleVector:
    total = max(len(examples), 1)
    captures = checks = castles = promotions = forcing = early_queen = opening = 0
    for example in examples:
        board = board_from_fen(example.fen)
        move = chess.Move.from_uci(example.move_uci)
        piece = board.piece_at(move.from_square)
        is_capture = board.is_capture(move)
        captures += int(is_capture)
        castles += int(board.is_castling(move))
        promotions += int(move.promotion is not None)
        phase = _phase_name(board)
        board.push(move)
        checks += int(board.is_check())
        forcing += int(is_capture or board.is_check())
        early_queen += int(
            piece is not None and piece.piece_type == chess.QUEEN and example.fullmove_number <= 12
        )
        opening += int(phase == "opening")

    return StyleVector(
        capture_rate=captures / total,
        check_rate=checks / total,
        castle_rate=castles / total,
        promotion_rate=promotions / total,
        forcing_rate=forcing / total,
        early_queen_rate=early_queen / total,
        opening_phase_rate=opening / total,
    )


def style_similarity(actual: StyleVector, predicted: StyleVector) -> float:
    actual_values = _style_values(actual)
    predicted_values = _style_values(predicted)
    distance = sum(
        abs(left - right) for left, right in zip(actual_values, predicted_values, strict=True)
    )
    return max(0.0, 1.0 - distance / len(actual_values))


def _style_values(style: StyleVector) -> tuple[float, ...]:
    return (
        style.capture_rate,
        style.check_rate,
        style.castle_rate,
        style.promotion_rate,
        style.forcing_rate,
        style.early_queen_rate,
        style.opening_phase_rate,
    )


def opening_similarity(actual: list[MoveExample], predicted: list[MoveExample]) -> float:
    return _histogram_overlap(_opening_counter(actual), _opening_counter(predicted))


def distribution_similarity(
    actual: Counter[str],
    predicted: Counter[str],
) -> DistributionSimilarityMetrics:
    keys = sorted(set(actual) | set(predicted))
    if not keys:
        return DistributionSimilarityMetrics(0.0, 0.0, None, None, "builtin")

    actual_vector = _normalized_vector(actual, keys)
    predicted_vector = _normalized_vector(predicted, keys)
    scipy_available = find_spec("scipy") is not None
    js_distance = (
        _scipy_jensen_shannon(actual_vector, predicted_vector) if scipy_available else None
    )
    wasserstein = _scipy_wasserstein(actual_vector, predicted_vector) if scipy_available else None
    return DistributionSimilarityMetrics(
        histogram_overlap=_histogram_overlap(actual, predicted),
        cosine_similarity=_cosine_similarity(actual_vector, predicted_vector),
        jensen_shannon_distance=js_distance,
        wasserstein_distance=wasserstein,
        backend="scipy" if scipy_available else "builtin",
    )


def prediction_confidence(
    model: PersonaModel,
    examples: list[MoveExample],
    *,
    top_k: int = 3,
) -> PredictionConfidenceMetrics:
    if not examples:
        return PredictionConfidenceMetrics(0, 0.0, 0.0, 0.0, 0.0)

    top_scores: list[float] = []
    correct_scores: list[float] = []
    incorrect_scores: list[float] = []
    margins: list[float] = []
    for example in examples:
        predictions = model.predict(board_from_fen(example.fen), top_k=top_k)
        if not predictions:
            continue
        top_score = predictions[0].score
        second_score = predictions[1].score if len(predictions) > 1 else 0.0
        top_scores.append(top_score)
        margins.append(top_score - second_score)
        if predictions[0].move_uci == example.move_uci:
            correct_scores.append(top_score)
        else:
            incorrect_scores.append(top_score)

    return PredictionConfidenceMetrics(
        examples=len(top_scores),
        average_top1_score=_mean(top_scores),
        average_correct_top1_score=_mean(correct_scores),
        average_incorrect_top1_score=_mean(incorrect_scores),
        average_top1_margin=_mean(margins),
    )


def evaluate_engine_quality(
    model: PersonaModel,
    examples: list[MoveExample],
    *,
    engine_path: str | Path,
    config: EngineGuidanceConfig,
    blunder_threshold_cp: int,
) -> EngineQualityMetrics:
    import chess.engine

    if not examples:
        return EngineQualityMetrics(0, 0.0, 0.0, 0.0, 0.0)

    model_scores: list[int] = []
    target_scores: list[int] = []
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    try:
        for example in examples:
            board = board_from_fen(example.fen)
            prediction = model.predict(board, top_k=1)
            if not prediction:
                continue
            model_move = chess.Move.from_uci(prediction[0].move_uci)
            target_move = chess.Move.from_uci(example.move_uci)
            if model_move not in board.legal_moves or target_move not in board.legal_moves:
                continue
            model_scores.append(_score_move(engine, board, model_move, config))
            target_scores.append(_score_move(engine, board, target_move, config))
    finally:
        engine.quit()

    if not model_scores:
        return EngineQualityMetrics(0, 0.0, 0.0, 0.0, 0.0)

    losses = [
        max(0, target - model) for model, target in zip(model_scores, target_scores, strict=True)
    ]
    return EngineQualityMetrics(
        examples=len(model_scores),
        average_model_centipawns=sum(model_scores) / len(model_scores),
        average_target_centipawns=sum(target_scores) / len(target_scores),
        average_centipawn_loss=sum(losses) / len(losses),
        blunder_rate=sum(loss >= blunder_threshold_cp for loss in losses) / len(losses),
    )


def _prediction_examples(model: PersonaModel, examples: list[MoveExample]) -> list[MoveExample]:
    predicted: list[MoveExample] = []
    for example in examples:
        board = board_from_fen(example.fen)
        predictions = model.predict(board, top_k=1)
        if not predictions:
            continue
        predicted.append(
            MoveExample(
                schema_version=example.schema_version,
                fen=example.fen,
                position_key=example.position_key,
                move_uci=predictions[0].move_uci,
                san=predictions[0].san,
                player=example.player,
                player_color=example.player_color,
                game_index=example.game_index,
                ply=example.ply,
                fullmove_number=example.fullmove_number,
                result=example.result,
                white=example.white,
                black=example.black,
            )
        )
    return predicted


def _segment_metrics(
    model: PersonaModel,
    examples: list[MoveExample],
    *,
    k: int,
    segment_type: str,
) -> list[SegmentEvaluation]:
    grouped: dict[str, list[MoveExample]] = {}
    for example in examples:
        board = board_from_fen(example.fen)
        key = _phase_name(board) if segment_type == "phase" else _piece_name(board, example)
        grouped.setdefault(key, []).append(example)
    return [
        SegmentEvaluation(
            name=name,
            segment_type=segment_type,
            examples=len(segment_examples),
            metrics=evaluate_move_matching(model, segment_examples, k=k),
        )
        for name, segment_examples in sorted(grouped.items())
    ]


def _top1_agreement(
    candidate_model: PersonaModel,
    baseline_model: PersonaModel,
    examples: list[MoveExample],
) -> float:
    if not examples:
        return 0.0
    matches = 0
    covered = 0
    for example in examples:
        board = board_from_fen(example.fen)
        candidate = candidate_model.predict(board, top_k=1)
        baseline = baseline_model.predict(board, top_k=1)
        if not candidate or not baseline:
            continue
        covered += 1
        matches += int(candidate[0].move_uci == baseline[0].move_uci)
    return matches / covered if covered else 0.0


def _score_move(
    engine: Any,
    board: chess.Board,
    move: chess.Move,
    config: EngineGuidanceConfig,
) -> int:
    child = board.copy(stack=False)
    child.push(move)
    info = engine.analyse(child, chess.engine.Limit(time=config.time_limit, depth=config.depth))
    return int(info["score"].pov(board.turn).score(mate_score=config.mate_score) or 0)


def _histogram_overlap(left: Counter[str], right: Counter[str]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 or right_total == 0:
        return 0.0
    keys = set(left) | set(right)
    return sum(min(left[key] / left_total, right[key] / right_total) for key in keys)


def _opening_counter(examples: list[MoveExample]) -> Counter[str]:
    return Counter(example.san for example in examples if example.fullmove_number <= 12)


def _style_counter(style: StyleVector) -> Counter[str]:
    return Counter({name: value for name, value in style.to_dict().items()})


def _normalized_vector(counter: Counter[str], keys: list[str]) -> list[float]:
    total = sum(counter.values())
    if total <= 0:
        return [0.0 for _ in keys]
    return [counter[key] / total for key in keys]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _scipy_jensen_shannon(left: list[float], right: list[float]) -> float:
    distance = import_module("scipy.spatial.distance")

    return float(distance.jensenshannon(left, right))


def _scipy_wasserstein(left: list[float], right: list[float]) -> float:
    stats = import_module("scipy.stats")

    return float(stats.wasserstein_distance(left, right))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _phase_name(board: chess.Board) -> str:
    non_pawn_material = _non_pawn_material(board)
    if board.fullmove_number <= 12 and non_pawn_material >= 44:
        return "opening"
    if non_pawn_material <= 18:
        return "endgame"
    return "middlegame"


def _piece_name(board: chess.Board, example: MoveExample) -> str:
    move = chess.Move.from_uci(example.move_uci)
    piece = board.piece_at(move.from_square)
    if piece is None:
        return "unknown"
    return {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }[piece.piece_type]


def _non_pawn_material(board: chess.Board) -> int:
    values = {
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in values.items()
        for color in (chess.WHITE, chess.BLACK)
    )
