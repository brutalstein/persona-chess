import json
from collections import Counter
from dataclasses import asdict, dataclass
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
class PersonaEvaluationReport:
    schema_version: str
    candidate_model_type: str
    examples: int
    move_matching: MoveMatchMetrics
    actual_style: StyleVector
    predicted_style: StyleVector
    style_similarity: float
    opening_similarity: float
    comparison: ModelComparisonMetrics | None = None
    engine_quality: EngineQualityMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["move_matching"] = self.move_matching.to_dict()
        data["actual_style"] = self.actual_style.to_dict()
        data["predicted_style"] = self.predicted_style.to_dict()
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
        opening_similarity=opening_similarity(examples, predicted_examples),
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
    actual_moves = Counter(example.san for example in actual if example.fullmove_number <= 12)
    predicted_moves = Counter(example.san for example in predicted if example.fullmove_number <= 12)
    return _histogram_overlap(actual_moves, predicted_moves)


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


def _phase_name(board: chess.Board) -> str:
    non_pawn_material = _non_pawn_material(board)
    if board.fullmove_number <= 12 and non_pawn_material >= 44:
        return "opening"
    if non_pawn_material <= 18:
        return "endgame"
    return "middlegame"


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
