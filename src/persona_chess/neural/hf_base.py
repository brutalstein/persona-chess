import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

import chess

from persona_chess.chess.legal import board_from_fen
from persona_chess.exceptions import OptionalDependencyError
from persona_chess.models.types import MovePrediction
from persona_chess.neural.cuda import resolve_torch_device, torch_runtime_info

DEFAULT_BASE_MODEL = "Maxlegrec/ChessBot"


def predict_hf_base_moves(
    fen: str,
    *,
    model_name: str = DEFAULT_BASE_MODEL,
    top_k: int = 8,
    device: str | None = None,
    temperature: float = 0.1,
) -> list[MovePrediction]:
    torch = _require_module("torch")
    board = board_from_fen(fen)
    active_device = resolve_torch_device(torch, requested_device=device)
    model = _load_hf_base_model(model_name, active_device)

    probabilities = model.get_move_from_fen_no_thinking(
        board.fen(),
        T=temperature,
        device=active_device,
        return_probs=True,
    )
    if not isinstance(probabilities, dict):
        return []

    legal_moves = {move.uci(): move for move in board.legal_moves}
    scored: list[tuple[chess.Move, float]] = []
    for move_uci, score in probabilities.items():
        move = legal_moves.get(str(move_uci))
        if move is not None:
            scored.append((move, float(score)))

    scored.sort(key=lambda item: (-item[1], item[0].uci()))
    return [
        MovePrediction.from_board(board, move=move, score=score, reason="hf_base_policy")
        for move, score in scored[:top_k]
    ]


def download_hf_base_model(
    model_name: str = DEFAULT_BASE_MODEL,
    *,
    device: str | None = None,
) -> Path | None:
    torch = _require_module("torch")
    active_device = resolve_torch_device(torch, requested_device=device)
    _load_hf_base_model(model_name, active_device)
    return None


@lru_cache(maxsize=4)
def _load_hf_base_model(model_name: str, device: str) -> Any:
    transformers = _require_module("transformers")
    torch = _require_module("torch")
    runtime = torch_runtime_info(torch, requested_device=device)
    print(
        "PersonaChess base model: "
        f"loading {model_name} on {runtime.selected_device} "
        f"(torch={runtime.torch_version}, cuda_build={runtime.cuda_build or 'cpu'})",
        file=sys.stderr,
        flush=True,
    )
    print(
        "PersonaChess base model: first load may download model weights from Hugging Face.",
        file=sys.stderr,
        flush=True,
    )
    model = transformers.AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype="auto",
    )
    model = model.to(device)
    model.eval()
    return model


def _require_module(name: str) -> Any:
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "Missing runtime dependency. Install persona-chess with: pip install persona-chess"
        ) from exc
