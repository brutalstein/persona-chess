from typing import Any

from persona_chess.exceptions import OptionalDependencyError
from persona_chess.neural.config import TransformerPolicyConfig
from persona_chess.neural.samples import PolicyBatch


def is_torch_available() -> bool:
    try:
        __import__("torch")
    except ModuleNotFoundError:
        return False
    return True


def require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "PyTorch is required for neural training. Install persona-chess with the ml extra."
        ) from exc
    return torch


def build_transformer_policy_model(
    *,
    config: TransformerPolicyConfig,
    position_vocabulary_size: int,
    move_vocabulary_size: int,
    pad_token_id: int = 0,
) -> Any:
    torch = require_torch()
    nn = torch.nn

    class TransformerPolicyModel(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(
                position_vocabulary_size,
                config.d_model,
                padding_idx=pad_token_id,
            )
            self.position_embedding = nn.Embedding(
                config.max_sequence_length,
                config.d_model,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
            self.policy_head = nn.Linear(config.d_model, move_vocabulary_size)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            batch_size, sequence_length = input_ids.shape
            positions = torch.arange(sequence_length, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, sequence_length)

            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
            encoded = self.encoder(hidden, src_key_padding_mask=attention_mask == 0)
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            pooled = encoded[torch.arange(batch_size, device=input_ids.device), lengths]
            return self.policy_head(pooled)

    return TransformerPolicyModel()


def policy_batch_to_tensors(batch: PolicyBatch, *, device: str | None = None) -> dict[str, Any]:
    torch = require_torch()
    target_device = torch.device(device) if device else None

    tensors = {
        "input_ids": torch.tensor(batch.input_ids, dtype=torch.long, device=target_device),
        "attention_mask": torch.tensor(
            batch.attention_mask,
            dtype=torch.long,
            device=target_device,
        ),
        "legal_move_ids": torch.tensor(
            batch.legal_move_ids, dtype=torch.long, device=target_device
        ),
        "legal_move_mask": torch.tensor(
            batch.legal_move_mask,
            dtype=torch.bool,
            device=target_device,
        ),
        "target_move_ids": torch.tensor(
            batch.target_move_ids,
            dtype=torch.long,
            device=target_device,
        ),
        "target_legal_indices": torch.tensor(
            batch.target_legal_indices,
            dtype=torch.long,
            device=target_device,
        ),
    }
    return tensors


def gather_legal_logits(logits: Any, legal_move_ids: Any, legal_move_mask: Any) -> Any:
    torch = require_torch()
    legal_logits = logits.gather(dim=1, index=legal_move_ids)
    return legal_logits.masked_fill(~legal_move_mask, torch.finfo(logits.dtype).min)
