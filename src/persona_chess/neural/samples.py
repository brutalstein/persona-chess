from dataclasses import asdict, dataclass
from typing import Any

from persona_chess.neural.position_vocabulary import PositionVocabulary
from persona_chess.neural.tokens import PositionTokenizer
from persona_chess.neural.vocabulary import MoveVocabulary
from persona_chess.training.records import TrainingRecord


@dataclass(frozen=True, slots=True)
class PolicySample:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    legal_move_ids: tuple[int, ...]
    target_legal_index: int
    target_move_id: int
    target_move: str
    fen: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    legal_move_ids: tuple[tuple[int, ...], ...]
    legal_move_mask: tuple[tuple[int, ...], ...]
    target_legal_indices: tuple[int, ...]
    target_move_ids: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.target_move_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_policy_sample(
    record: TrainingRecord,
    *,
    position_vocabulary: PositionVocabulary,
    move_vocabulary: MoveVocabulary,
    max_sequence_length: int,
    tokenizer: PositionTokenizer | None = None,
) -> PolicySample:
    active_tokenizer = tokenizer or PositionTokenizer()
    tokens = active_tokenizer.tokenize_record(record)
    input_ids = position_vocabulary.encode_tokens(tokens, max_length=max_sequence_length)
    attention_mask = tuple(
        0 if token_id == position_vocabulary.pad_id else 1 for token_id in input_ids
    )
    legal_move_ids = tuple(move_vocabulary.encode(move) for move in record.legal_moves)
    target_move_id = move_vocabulary.encode(record.target_move)
    target_legal_index = _target_legal_index(record)

    return PolicySample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        legal_move_ids=legal_move_ids,
        target_legal_index=target_legal_index,
        target_move_id=target_move_id,
        target_move=record.target_move,
        fen=record.fen,
    )


def build_policy_samples(
    records: list[TrainingRecord],
    *,
    position_vocabulary: PositionVocabulary,
    move_vocabulary: MoveVocabulary,
    max_sequence_length: int,
    tokenizer: PositionTokenizer | None = None,
) -> list[PolicySample]:
    return [
        build_policy_sample(
            record,
            position_vocabulary=position_vocabulary,
            move_vocabulary=move_vocabulary,
            max_sequence_length=max_sequence_length,
            tokenizer=tokenizer,
        )
        for record in records
    ]


def collate_policy_samples(
    samples: list[PolicySample],
    *,
    move_pad_id: int,
) -> PolicyBatch:
    if not samples:
        return PolicyBatch(
            input_ids=(),
            attention_mask=(),
            legal_move_ids=(),
            legal_move_mask=(),
            target_legal_indices=(),
            target_move_ids=(),
        )

    max_legal_moves = max(len(sample.legal_move_ids) for sample in samples)
    legal_move_ids = tuple(
        _pad_tuple(sample.legal_move_ids, max_legal_moves, move_pad_id) for sample in samples
    )
    legal_move_mask = tuple(
        _legal_mask(len(sample.legal_move_ids), max_legal_moves) for sample in samples
    )

    return PolicyBatch(
        input_ids=tuple(sample.input_ids for sample in samples),
        attention_mask=tuple(sample.attention_mask for sample in samples),
        legal_move_ids=legal_move_ids,
        legal_move_mask=legal_move_mask,
        target_legal_indices=tuple(sample.target_legal_index for sample in samples),
        target_move_ids=tuple(sample.target_move_id for sample in samples),
    )


def _pad_tuple(values: tuple[int, ...], length: int, pad_id: int) -> tuple[int, ...]:
    if len(values) >= length:
        return values
    return (*values, *(pad_id for _ in range(length - len(values))))


def _legal_mask(active_length: int, padded_length: int) -> tuple[int, ...]:
    return (*(1 for _ in range(active_length)), *(0 for _ in range(padded_length - active_length)))


def _target_legal_index(record: TrainingRecord) -> int:
    try:
        return record.legal_moves.index(record.target_move)
    except ValueError as exc:
        raise ValueError(f"Target move is not legal in record: {record.target_move}") from exc
