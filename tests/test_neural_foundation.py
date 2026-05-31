from pathlib import Path

import pytest

from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset.builder import build_move_examples
from persona_chess.exceptions import OptionalDependencyError
from persona_chess.neural import (
    AdapterManifest,
    LoraConfig,
    MoveVocabulary,
    NeuralCheckpointManifest,
    NeuralTrainingConfig,
    PositionTokenizer,
    PositionVocabulary,
    TrainingResult,
    TransformerPolicyConfig,
    apply_lora_adapter,
    build_policy_samples,
    build_transformer_policy_model,
    collate_policy_samples,
    create_adapter_manifest,
    create_adapter_manifest_from_vocabulary_sizes,
    is_peft_available,
    is_torch_available,
    iter_policy_batches,
    legal_move_id_entries,
    predict_policy_moves_from_checkpoint,
    prepare_streaming_neural_artifacts,
    save_torch_policy_checkpoint,
    summarize_trainable_parameters,
    train_policy_model,
    validate_neural_artifacts,
)
from persona_chess.pgn.filters import GameFilter
from persona_chess.training import build_training_records

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_move_vocabulary_round_trips_records(tmp_path: Path) -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    vocabulary = MoveVocabulary.from_records(records)

    assert vocabulary.size > 4000
    assert vocabulary.decode(vocabulary.encode("e2e4")) == "e2e4"
    assert vocabulary.decode(vocabulary.encode("not-a-move")) == "<move-unk>"

    output = tmp_path / "moves.json"
    vocabulary.save(output)

    assert MoveVocabulary.load(output) == vocabulary


def test_standard_move_vocabulary_covers_start_position_legal_moves() -> None:
    vocabulary = MoveVocabulary.standard()
    board = board_from_fen("startpos")
    entries = legal_move_id_entries(board, vocabulary)

    assert len(entries) == len(list(board.legal_moves))
    assert all(move_id != vocabulary.unk_id for _, move_id in entries)


def test_adapter_manifest_captures_neural_training_plan(tmp_path: Path) -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    manifest = create_adapter_manifest(
        records,
        player="Target Player",
        transformer=TransformerPolicyConfig(d_model=128, n_heads=4),
        lora=LoraConfig(rank=4, alpha=8),
    )

    assert manifest.schema_version == "persona-chess/neural-adapter-manifest/v1"
    assert manifest.training_examples == len(records)
    assert manifest.move_vocabulary_size == MoveVocabulary.from_records(records).size
    assert manifest.position_vocabulary_size == PositionVocabulary.from_records(records).size

    output = tmp_path / "manifest.json"
    manifest.save(output)

    assert AdapterManifest.load(output).to_dict() == manifest.to_dict()


def test_position_tokenizer_and_vocabulary_round_trip(tmp_path: Path) -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    tokenizer = PositionTokenizer()
    vocabulary = PositionVocabulary.from_records(records, tokenizer=tokenizer)

    tokens = tokenizer.tokenize_record(records[0])
    encoded = vocabulary.encode_tokens(tokens, max_length=32)

    assert tokens[0] == "<bos>"
    assert "turn:w" in tokens
    assert vocabulary.decode(encoded[0]) == "<bos>"
    assert len(encoded) == 32

    output = tmp_path / "positions.json"
    vocabulary.save(output)

    assert PositionVocabulary.load(output) == vocabulary


def test_policy_samples_and_batches_are_model_ready() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    position_vocabulary = PositionVocabulary.from_records(records)
    move_vocabulary = MoveVocabulary.from_records(records)

    samples = build_policy_samples(
        records[:2],
        position_vocabulary=position_vocabulary,
        move_vocabulary=move_vocabulary,
        max_sequence_length=32,
    )
    batch = collate_policy_samples(samples, move_pad_id=move_vocabulary.pad_id)

    assert batch.size == 2
    assert len(batch.input_ids[0]) == 32
    assert len(batch.legal_move_ids[0]) == len(batch.legal_move_ids[1])
    assert batch.target_legal_indices[0] == records[0].target_index
    assert samples[0].target_move_id == move_vocabulary.encode(records[0].target_move)


def test_streaming_policy_batches_are_model_ready() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    artifacts = prepare_streaming_neural_artifacts(iter(records))

    batches = list(
        iter_policy_batches(
            iter(records),
            position_vocabulary=artifacts.position_vocabulary,
            move_vocabulary=artifacts.move_vocabulary,
            max_sequence_length=32,
            batch_size=3,
        )
    )

    assert artifacts.training_examples == len(records)
    assert artifacts.move_vocabulary.size > 4000
    assert sum(batch.size for batch in batches) == len(records)
    assert batches[0].size == 3


def test_torch_training_reports_validation_metrics() -> None:
    if not is_torch_available():
        pytest.skip("PyTorch is not installed in this environment.")

    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    transformer = TransformerPolicyConfig(
        max_sequence_length=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    training = NeuralTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        warmup_ratio=0.0,
        mixed_precision="off",
    )
    move_vocabulary = MoveVocabulary.from_records(records)
    position_vocabulary = PositionVocabulary.from_records(records)
    batches = list(
        iter_policy_batches(
            iter(records[:8]),
            position_vocabulary=position_vocabulary,
            move_vocabulary=move_vocabulary,
            max_sequence_length=transformer.max_sequence_length,
            batch_size=training.batch_size,
        )
    )
    validation_batches = list(
        iter_policy_batches(
            iter(records[8:]),
            position_vocabulary=position_vocabulary,
            move_vocabulary=move_vocabulary,
            max_sequence_length=transformer.max_sequence_length,
            batch_size=training.batch_size,
        )
    )

    _, result = train_policy_model(
        batches,
        transformer=transformer,
        training=training,
        position_vocabulary_size=position_vocabulary.size,
        move_vocabulary_size=move_vocabulary.size,
        validation_batches=validation_batches,
        device="cpu",
    )

    assert result.optimizer_steps > 0
    assert result.average_train_loss > 0
    assert result.validation_examples == 2
    assert result.validation_loss is not None
    assert result.validation_accuracy is not None
    assert result.validation_top3_accuracy is not None
    assert result.best_epoch == 1
    assert result.mixed_precision == "off"


def test_create_adapter_manifest_from_streaming_artifact_sizes() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    artifacts = prepare_streaming_neural_artifacts(iter(records))
    manifest = create_adapter_manifest_from_vocabulary_sizes(
        player="Target Player",
        move_vocabulary_size=artifacts.move_vocabulary.size,
        position_vocabulary_size=artifacts.position_vocabulary.size,
        training_examples=artifacts.training_examples,
    )

    assert manifest.move_vocabulary_size == artifacts.move_vocabulary.size
    assert manifest.position_vocabulary_size == artifacts.position_vocabulary.size
    assert manifest.training_examples == len(records)


def test_adapter_manifest_captures_both_vocabularies() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    manifest = create_adapter_manifest(records, player="Target Player")

    assert manifest.move_vocabulary_size == MoveVocabulary.from_records(records).size
    assert manifest.position_vocabulary_size == PositionVocabulary.from_records(records).size


def test_neural_artifact_validation_detects_size_mismatch() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    manifest = create_adapter_manifest(records, player="Target Player")
    move_vocabulary = MoveVocabulary.from_records(records)
    position_vocabulary = PositionVocabulary.from_records(records)

    validation = validate_neural_artifacts(
        manifest=manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
    )

    assert validation.ok
    assert validation.errors == ()

    bad_manifest = AdapterManifest.create(
        base_model=manifest.base_model,
        player=manifest.player,
        transformer=manifest.transformer,
        lora=manifest.lora,
        training=manifest.training,
        move_vocabulary_size=manifest.move_vocabulary_size + 1,
        position_vocabulary_size=manifest.position_vocabulary_size,
        training_examples=manifest.training_examples,
    )
    bad_validation = validate_neural_artifacts(
        manifest=bad_manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
    )

    assert not bad_validation.ok
    assert "move_vocabulary_size mismatch" in bad_validation.errors[0]


def test_neural_checkpoint_manifest_round_trip(tmp_path: Path) -> None:
    manifest = NeuralCheckpointManifest.create(
        player="Target Player",
        training_result=TrainingResult(
            epochs=1,
            steps=2,
            final_loss=0.25,
            trainable_parameters=10,
            total_parameters=100,
        ),
        lora_applied=True,
    )
    output = tmp_path / "checkpoint.json"

    manifest.save(output)
    loaded = NeuralCheckpointManifest.load(output)

    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.lora_applied


def test_lora_config_targets_current_torch_backend_modules() -> None:
    assert LoraConfig().target_modules == ("out_proj", "linear1", "linear2")


def test_peft_lora_backend_is_optional() -> None:
    if is_peft_available():
        pytest.skip("PEFT is installed in this environment.")

    with pytest.raises(OptionalDependencyError):
        apply_lora_adapter(object(), LoraConfig())


def test_trainable_parameter_summary_handles_simple_model() -> None:
    class Parameter:
        def __init__(self, count: int, requires_grad: bool) -> None:
            self.requires_grad = requires_grad
            self._count = count

        def numel(self) -> int:
            return self._count

    class Model:
        def parameters(self) -> list[Parameter]:
            return [Parameter(10, True), Parameter(30, False)]

    summary = summarize_trainable_parameters(Model())

    assert summary.trainable_parameters == 10
    assert summary.total_parameters == 40
    assert summary.trainable_ratio == 0.25


def test_torch_backend_is_optional() -> None:
    config = TransformerPolicyConfig(d_model=32, n_heads=4, n_layers=1)

    if is_torch_available():
        model = build_transformer_policy_model(
            config=config,
            position_vocabulary_size=32,
            move_vocabulary_size=64,
        )
        assert model is not None
        return

    with pytest.raises(OptionalDependencyError):
        build_transformer_policy_model(
            config=config,
            position_vocabulary_size=32,
            move_vocabulary_size=64,
        )


def test_torch_checkpoint_inference_returns_legal_predictions(tmp_path: Path) -> None:
    if not is_torch_available():
        pytest.skip("PyTorch is not installed in this environment.")

    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    transformer = TransformerPolicyConfig(
        max_sequence_length=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
    )
    move_vocabulary = MoveVocabulary.from_records(records)
    position_vocabulary = PositionVocabulary.from_records(records)
    manifest = create_adapter_manifest(
        records,
        player="Target Player",
        transformer=transformer,
    )
    model = build_transformer_policy_model(
        config=transformer,
        position_vocabulary_size=position_vocabulary.size,
        move_vocabulary_size=move_vocabulary.size,
        pad_token_id=position_vocabulary.pad_id,
    )

    save_torch_policy_checkpoint(
        tmp_path,
        model=model,
        adapter_manifest=manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
    )
    predictions = predict_policy_moves_from_checkpoint(tmp_path, fen="startpos", top_k=3)
    legal_uci = {move.uci() for move in board_from_fen("startpos").legal_moves}

    assert len(predictions) == 3
    assert {prediction.move_uci for prediction in predictions} <= legal_uci
    assert all(prediction.reason == "neural_policy" for prediction in predictions)


def test_peft_lora_adapter_smoke_when_available() -> None:
    if not is_torch_available() or not is_peft_available():
        pytest.skip("PyTorch and PEFT are not installed in this environment.")

    model = build_transformer_policy_model(
        config=TransformerPolicyConfig(d_model=32, n_heads=4, n_layers=1),
        position_vocabulary_size=32,
        move_vocabulary_size=64,
    )
    adapted_model, summary = apply_lora_adapter(model, LoraConfig(rank=2, alpha=4))

    assert adapted_model is not None
    assert summary.trainable_parameters > 0
    assert summary.trainable_parameters < summary.total_parameters


def test_neural_checkpoint_prediction_requires_torch_when_unavailable(tmp_path: Path) -> None:
    if is_torch_available():
        pytest.skip("PyTorch is installed in this environment.")

    with pytest.raises(OptionalDependencyError):
        predict_policy_moves_from_checkpoint(tmp_path, fen="startpos")


def test_neural_configs_validate_values() -> None:
    with pytest.raises(ValueError):
        TransformerPolicyConfig(d_model=33, n_heads=8)

    with pytest.raises(ValueError):
        NeuralTrainingConfig(batch_size=0)


def test_neural_training_config_loads_legacy_payloads() -> None:
    config = NeuralTrainingConfig.from_dict(
        {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "seed": 7,
        }
    )

    assert config.gradient_accumulation_steps == 1
    assert config.max_grad_norm == 1.0
    assert config.mixed_precision == "auto"
