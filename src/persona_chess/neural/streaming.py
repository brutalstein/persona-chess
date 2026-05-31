from collections.abc import Iterable
from dataclasses import dataclass

from persona_chess.neural.position_vocabulary import PositionVocabulary
from persona_chess.neural.tokens import PositionTokenizer
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
    standard_position_vocabulary: bool = True,
) -> StreamingNeuralArtifacts:
    active_tokenizer = tokenizer or PositionTokenizer()
    tokens: set[str] = set()
    training_examples = 0

    for record in records:
        training_examples += 1
        if not standard_position_vocabulary:
            tokens.update(active_tokenizer.tokenize_record(record))

    position_vocabulary = (
        PositionVocabulary.standard(tokenizer=active_tokenizer)
        if standard_position_vocabulary
        else PositionVocabulary.from_tokens(tokens)
    )
    return StreamingNeuralArtifacts(
        move_vocabulary=MoveVocabulary.standard(),
        position_vocabulary=position_vocabulary,
        training_examples=training_examples,
    )
