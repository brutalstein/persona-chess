from persona_chess.neural.config import LoraConfig, NeuralTrainingConfig, TransformerPolicyConfig
from persona_chess.neural.manifest import AdapterManifest
from persona_chess.neural.position_vocabulary import PositionVocabulary
from persona_chess.neural.tokens import PositionTokenizer
from persona_chess.neural.vocabulary import MoveVocabulary
from persona_chess.training.records import TrainingRecord


def create_adapter_manifest(
    records: list[TrainingRecord],
    *,
    player: str,
    base_model: str = "persona-chess/base-small",
    transformer: TransformerPolicyConfig | None = None,
    lora: LoraConfig | None = None,
    training: NeuralTrainingConfig | None = None,
    tokenizer: PositionTokenizer | None = None,
) -> AdapterManifest:
    vocabulary = MoveVocabulary.from_records(records)
    position_vocabulary = PositionVocabulary.from_records(records, tokenizer=tokenizer)
    return AdapterManifest.create(
        base_model=base_model,
        player=player,
        transformer=transformer or TransformerPolicyConfig(),
        lora=lora or LoraConfig(),
        training=training or NeuralTrainingConfig(),
        move_vocabulary_size=vocabulary.size,
        position_vocabulary_size=position_vocabulary.size,
        training_examples=len(records),
    )


def create_adapter_manifest_from_vocabulary_sizes(
    *,
    player: str,
    move_vocabulary_size: int,
    position_vocabulary_size: int,
    training_examples: int,
    base_model: str = "persona-chess/base-small",
    transformer: TransformerPolicyConfig | None = None,
    lora: LoraConfig | None = None,
    training: NeuralTrainingConfig | None = None,
) -> AdapterManifest:
    return AdapterManifest.create(
        base_model=base_model,
        player=player,
        transformer=transformer or TransformerPolicyConfig(),
        lora=lora or LoraConfig(),
        training=training or NeuralTrainingConfig(),
        move_vocabulary_size=move_vocabulary_size,
        position_vocabulary_size=position_vocabulary_size,
        training_examples=training_examples,
    )
