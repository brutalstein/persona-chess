from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess

from persona_chess.chess.legal import board_from_fen, sorted_legal_moves
from persona_chess.models.types import MovePrediction
from persona_chess.neural.checkpoint import load_torch_policy_checkpoint
from persona_chess.neural.config import TransformerPolicyConfig
from persona_chess.neural.position_vocabulary import PositionVocabulary
from persona_chess.neural.tokens import PositionTokenizer
from persona_chess.neural.torch_backend import require_torch
from persona_chess.neural.vocabulary import MoveVocabulary


@dataclass(frozen=True, slots=True)
class NeuralPredictionTrace:
    legal_moves: int
    scored_legal_moves: int
    unknown_legal_moves: int
    device: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_policy_moves_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    fen: str,
    top_k: int = 3,
    device: str | None = None,
) -> list[MovePrediction]:
    model, _, adapter_manifest, move_vocabulary, position_vocabulary = load_torch_policy_checkpoint(
        checkpoint_dir,
        device=device,
    )
    return predict_policy_moves(
        model,
        fen=fen,
        transformer=adapter_manifest.transformer,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
        top_k=top_k,
        device=device,
    )


def predict_policy_moves(
    model: Any,
    *,
    fen: str,
    transformer: TransformerPolicyConfig,
    move_vocabulary: MoveVocabulary,
    position_vocabulary: PositionVocabulary,
    top_k: int = 3,
    device: str | None = None,
    tokenizer: PositionTokenizer | None = None,
) -> list[MovePrediction]:
    torch = require_torch()
    board = board_from_fen(fen)
    legal_entries = legal_move_id_entries(board, move_vocabulary)
    if not legal_entries:
        return _uniform_fallback(board, top_k=top_k, reason="neural_unknown_fallback")

    active_tokenizer = tokenizer or PositionTokenizer()
    input_ids = position_vocabulary.encode_tokens(
        active_tokenizer.tokenize_fen(fen),
        max_length=transformer.max_sequence_length,
    )
    attention_mask = tuple(
        0 if token_id == position_vocabulary.pad_id else 1 for token_id in input_ids
    )
    target_device = torch.device(device) if device else None

    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor([input_ids], dtype=torch.long, device=target_device),
            torch.tensor([attention_mask], dtype=torch.long, device=target_device),
        )[0]
        legal_move_ids = torch.tensor(
            [move_id for _, move_id in legal_entries],
            dtype=torch.long,
            device=target_device,
        )
        legal_logits = logits.gather(dim=0, index=legal_move_ids)
        probabilities = torch.softmax(legal_logits, dim=0).detach().cpu().tolist()

    scored = [
        (move, float(probability))
        for (move, _), probability in zip(legal_entries, probabilities, strict=True)
    ]
    scored.sort(key=lambda item: (-item[1], item[0].uci()))
    return [
        MovePrediction.from_board(board, move=move, score=score, reason="neural_policy")
        for move, score in scored[:top_k]
    ]


def legal_move_id_entries(
    board: chess.Board,
    move_vocabulary: MoveVocabulary,
) -> list[tuple[chess.Move, int]]:
    entries: list[tuple[chess.Move, int]] = []
    for move in sorted_legal_moves(board):
        move_id = move_vocabulary.encode(move.uci())
        if move_id != move_vocabulary.unk_id:
            entries.append((move, move_id))
    return entries


def _uniform_fallback(board: chess.Board, *, top_k: int, reason: str) -> list[MovePrediction]:
    legal_moves = sorted_legal_moves(board)
    if not legal_moves:
        return []

    score = 1 / len(legal_moves)
    return [
        MovePrediction.from_board(board, move=move, score=score, reason=reason)
        for move in legal_moves[:top_k]
    ]
