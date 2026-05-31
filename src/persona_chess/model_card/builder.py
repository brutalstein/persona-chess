from collections import Counter
from pathlib import Path

import chess

from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset.builder import build_move_examples
from persona_chess.dataset.records import MoveExample
from persona_chess.model_card.types import (
    DataQualitySummary,
    ModelCardRecommendation,
    MoveBreakdown,
    PersonaModelCard,
    StyleSummary,
)
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.profile.builder import build_profile
from persona_chess.profile.types import PersonaProfile


def build_model_card(
    path: str | Path,
    *,
    player: str,
    color: PlayerColor = "both",
    max_games: int | None = None,
    skip_first_plies: int = 0,
) -> PersonaModelCard:
    game_filter = GameFilter(player=player, color=color, max_games=max_games)
    examples = build_move_examples(path, game_filter, skip_first_plies=skip_first_plies)
    profile = build_profile(path, game_filter)
    breakdown = _build_move_breakdown(examples)
    style = _build_style_summary(examples, profile=profile)
    data_quality = _build_data_quality(examples, games=profile.games)
    recommendation = _build_recommendation(data_quality, style)

    return PersonaModelCard.create(
        profile=profile,
        data_quality=data_quality,
        style=style,
        move_breakdown=breakdown,
        recommendation=recommendation,
    )


def _build_data_quality(examples: list[MoveExample], *, games: int) -> DataQualitySummary:
    example_count = len(examples)
    unique_positions = len({example.position_key for example in examples})
    duplicate_position_rate = 1 - (unique_positions / example_count) if example_count else 0.0
    legal_move_counts = [len(list(board_from_fen(example.fen).legal_moves)) for example in examples]
    average_legal_moves = (
        sum(legal_move_counts) / len(legal_move_counts) if legal_move_counts else 0.0
    )
    warnings = _data_quality_warnings(
        games=games,
        examples=example_count,
        duplicate_position_rate=duplicate_position_rate,
    )

    return DataQualitySummary(
        games=games,
        examples=example_count,
        unique_positions=unique_positions,
        duplicate_position_rate=duplicate_position_rate,
        average_legal_moves=average_legal_moves,
        confidence=_confidence_label(games=games, examples=example_count),
        warnings=tuple(warnings),
    )


def _build_style_summary(
    examples: list[MoveExample],
    *,
    profile: PersonaProfile,
) -> StyleSummary:
    total = max(len(examples), 1)
    forcing_moves = 0
    early_queen_moves = 0
    opening_phase_moves = 0
    fullmove_total = 0

    for example in examples:
        board = board_from_fen(example.fen)
        move = chess.Move.from_uci(example.move_uci)
        piece = board.piece_at(move.from_square)
        is_capture = board.is_capture(move)
        phase = _phase_name(board)
        board.push(move)
        if is_capture or board.is_check():
            forcing_moves += 1

        if piece is not None and piece.piece_type == chess.QUEEN and example.fullmove_number <= 12:
            early_queen_moves += 1
        if phase == "opening":
            opening_phase_moves += 1
        fullmove_total += example.fullmove_number

    tendencies = profile.tendencies
    capture_rate = tendencies.capture_rate
    check_rate = tendencies.check_rate
    castle_rate = tendencies.kingside_castle_rate + tendencies.queenside_castle_rate
    forcing_rate = min(1.0, forcing_moves / total)
    early_queen_rate = early_queen_moves / total
    opening_phase_rate = opening_phase_moves / total
    average_fullmove_number = fullmove_total / total

    return StyleSummary(
        tags=tuple(
            _style_tags(
                capture_rate=capture_rate,
                check_rate=check_rate,
                castle_rate=castle_rate,
                early_queen_rate=early_queen_rate,
                opening_phase_rate=opening_phase_rate,
            )
        ),
        forcing_rate=forcing_rate,
        capture_rate=capture_rate,
        check_rate=check_rate,
        castle_rate=castle_rate,
        early_queen_rate=early_queen_rate,
        opening_phase_rate=opening_phase_rate,
        average_fullmove_number=average_fullmove_number,
    )


def _build_move_breakdown(examples: list[MoveExample]) -> MoveBreakdown:
    phase_counts: Counter[str] = Counter()
    piece_counts: Counter[str] = Counter()

    for example in examples:
        board = board_from_fen(example.fen)
        move = chess.Move.from_uci(example.move_uci)
        phase_counts[_phase_name(board)] += 1
        piece = board.piece_at(move.from_square)
        piece_counts[_piece_name(piece)] += 1

    total = max(len(examples), 1)
    return MoveBreakdown(
        phase_distribution=dict(phase_counts.most_common()),
        piece_distribution=dict(piece_counts.most_common()),
        piece_rates={piece: count / total for piece, count in sorted(piece_counts.items())},
    )


def _build_recommendation(
    data_quality: DataQualitySummary,
    style: StyleSummary,
) -> ModelCardRecommendation:
    notes: list[str] = []
    recommended_model = "blend"
    recommended_inference = "baseline"
    neural_readiness = "not_ready"

    if data_quality.examples >= 500 and data_quality.games >= 30:
        recommended_model = "neural_lora"
        recommended_inference = "neural_checkpoint"
        neural_readiness = "adapter_candidate"
        notes.append("Dataset size is large enough to justify LoRA adapter experiments.")
    elif data_quality.examples >= 60 and data_quality.games >= 8:
        recommended_inference = "engine_guided"
        neural_readiness = "warmup_candidate"
        notes.append("Use blend or engine-guided inference before investing in LoRA training.")
    else:
        notes.append("Collect more games before expecting stable player-style generalization.")

    if style.castle_rate < 0.05:
        notes.append("Castling rate is low; verify that the sample contains complete games.")
    if data_quality.duplicate_position_rate > 0.25:
        notes.append("Repeated positions are useful for memory models but can inflate confidence.")

    return ModelCardRecommendation(
        recommended_model=recommended_model,
        recommended_inference=recommended_inference,
        neural_readiness=neural_readiness,
        notes=tuple(notes),
    )


def _data_quality_warnings(
    *,
    games: int,
    examples: int,
    duplicate_position_rate: float,
) -> list[str]:
    warnings: list[str] = []
    if games < 5:
        warnings.append("Very small game sample; treat style tags as directional.")
    if examples < 60:
        warnings.append(
            "Low move count; baseline persona models are more reliable than neural adapters."
        )
    if duplicate_position_rate > 0.35:
        warnings.append("High repeated-position rate; held-out evaluation is especially important.")
    return warnings


def _confidence_label(*, games: int, examples: int) -> str:
    if games >= 30 and examples >= 500:
        return "high"
    if games >= 8 and examples >= 120:
        return "medium"
    return "low"


def _style_tags(
    *,
    capture_rate: float,
    check_rate: float,
    castle_rate: float,
    early_queen_rate: float,
    opening_phase_rate: float,
) -> list[str]:
    tags: list[str] = []
    if capture_rate >= 0.35:
        tags.append("tactical")
    elif capture_rate <= 0.18:
        tags.append("positional")

    if check_rate >= 0.18:
        tags.append("forcing")
    if castle_rate >= 0.12:
        tags.append("king-safety-aware")
    if early_queen_rate >= 0.08:
        tags.append("early-queen-activity")
    if opening_phase_rate >= 0.50:
        tags.append("opening-heavy-sample")

    if not tags:
        tags.append("balanced")
    return tags


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


def _piece_name(piece: chess.Piece | None) -> str:
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
