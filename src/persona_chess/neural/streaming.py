from collections.abc import Iterable
from dataclasses import dataclass

from persona_chess.neural.position_vocabulary import POSITION_VOCABULARY_SCHEMA, PositionVocabulary
from persona_chess.neural.tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    PositionTokenizer,
)
from persona_chess.neural.vocabulary import MoveVocabulary
from persona_chess.training.records import TrainingRecord


@dataclass(frozen=True, slots=True)
class StreamingNeuralArtifacts:
    move_vocabulary: MoveVocabulary
    position_vocabulary: PositionVocabulary
    training_examples: int


def prepare_streaming_neural_artifacts(
    records: Iterable[TrainingRecord],
    *,
    tokenizer: PositionTokenizer | None = None,
) -> StreamingNeuralArtifacts:
    active_tokenizer = tokenizer or PositionTokenizer()
    tokens: set[str] = set()
    training_examples = 0

    for record in records:
        training_examples += 1
        tokens.update(active_tokenizer.tokenize_record(record))

    special_tokens = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
    body_tokens = sorted(tokens.difference(special_tokens))
    return StreamingNeuralArtifacts(
        move_vocabulary=MoveVocabulary.standard(),
        position_vocabulary=PositionVocabulary(
            schema_version=POSITION_VOCABULARY_SCHEMA,
            id_to_token=(*special_tokens, *body_tokens),
        ),
        training_examples=training_examples,
    )
