import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persona_chess.exceptions import ArtifactError
from persona_chess.neural.tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    PositionTokenizer,
)
from persona_chess.training.records import TrainingRecord

POSITION_VOCABULARY_SCHEMA = "persona-chess/position-vocabulary/v1"


@dataclass(frozen=True, slots=True)
class PositionVocabulary:
    schema_version: str
    id_to_token: tuple[str, ...]

    @classmethod
    def from_records(
        cls,
        records: Iterable[TrainingRecord],
        *,
        tokenizer: PositionTokenizer | None = None,
    ) -> "PositionVocabulary":
        active_tokenizer = tokenizer or PositionTokenizer()
        tokens = {token for record in records for token in active_tokenizer.tokenize_record(record)}
        return cls.from_tokens(tokens)

    @classmethod
    def from_tokens(cls, tokens: Iterable[str]) -> "PositionVocabulary":
        special_tokens = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
        body_tokens = sorted(set(tokens).difference(special_tokens))
        return cls(
            schema_version=POSITION_VOCABULARY_SCHEMA,
            id_to_token=(*special_tokens, *body_tokens),
        )

    @classmethod
    def standard(cls, *, tokenizer: PositionTokenizer | None = None) -> "PositionVocabulary":
        active_tokenizer = tokenizer or PositionTokenizer()
        special_tokens = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
        body_tokens = {
            *_turn_tokens(),
            *_castling_tokens(),
            *_ep_tokens(),
            *_phase_tokens(),
            *_piece_square_tokens(),
        }
        if active_tokenizer.include_legal_moves:
            body_tokens.update(_legal_move_tokens())
        return cls(
            schema_version=POSITION_VOCABULARY_SCHEMA,
            id_to_token=(*special_tokens, *sorted(body_tokens)),
        )

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def encode(self, token: str) -> int:
        return self.token_to_id().get(token, self.unk_id)

    def encode_tokens(self, tokens: tuple[str, ...], *, max_length: int) -> tuple[int, ...]:
        encoded = tuple(self.encode(token) for token in tokens[:max_length])
        if len(encoded) >= max_length:
            return encoded
        return (*encoded, *(self.pad_id for _ in range(max_length - len(encoded))))

    def decode(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.size:
            return UNK_TOKEN
        return self.id_to_token[token_id]

    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id_to_token": list(self.id_to_token),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionVocabulary":
        if data.get("schema_version") != POSITION_VOCABULARY_SCHEMA:
            raise ArtifactError(
                f"Unsupported position vocabulary schema: {data.get('schema_version')}"
            )
        return cls(
            schema_version=data["schema_version"],
            id_to_token=tuple(data["id_to_token"]),
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        try:
            output_path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(f"Unable to save position vocabulary: {output_path}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "PositionVocabulary":
        input_path = Path(path)
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Unable to load position vocabulary: {input_path}") from exc
        return cls.from_dict(data)


def _turn_tokens() -> set[str]:
    return {"turn:w", "turn:b"}


def _castling_tokens() -> set[str]:
    rights = "KQkq"
    tokens = {"castling:-"}
    for mask in range(1, 1 << len(rights)):
        tokens.add(
            "castling:"
            + "".join(right for index, right in enumerate(rights) if mask & (1 << index))
        )
    return tokens


def _ep_tokens() -> set[str]:
    return {"ep:-", *(f"ep:{square}" for square in _square_names())}


def _phase_tokens() -> set[str]:
    return {"phase:opening", "phase:middlegame", "phase:endgame"}


def _piece_square_tokens() -> set[str]:
    pieces = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
    return {f"{piece}@{square}" for piece in pieces for square in _square_names()}


def _legal_move_tokens() -> set[str]:
    moves = {
        f"legal:{from_square}{to_square}"
        for from_square in _square_names()
        for to_square in _square_names()
        if from_square != to_square
    }
    moves.update(f"legal:{move}" for move in _promotion_moves())
    return moves


def _promotion_moves() -> set[str]:
    moves: set[str] = set()
    promotion_pieces = ("q", "r", "b", "n")

    for file in "abcdefgh":
        targets = _promotion_targets(file)
        moves.update(f"{file}7{target}8{piece}" for target in targets for piece in promotion_pieces)
        moves.update(f"{file}2{target}1{piece}" for target in targets for piece in promotion_pieces)
    return moves


def _promotion_targets(file: str) -> tuple[str, ...]:
    files = "abcdefgh"
    index = files.index(file)
    return tuple(
        files[target] for target in (index - 1, index, index + 1) if 0 <= target < len(files)
    )


def _square_names() -> tuple[str, ...]:
    files = "abcdefgh"
    ranks = "12345678"
    return tuple(f"{file}{rank}" for rank in ranks for file in files)
