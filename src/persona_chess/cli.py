import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import chess.engine
import typer

from persona_chess import PersonaChess
from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset import SplitConfig, split_examples
from persona_chess.dataset.builder import build_move_examples, iter_move_examples
from persona_chess.dataset.jsonl import write_examples_jsonl
from persona_chess.dataset.records import MoveExample
from persona_chess.engines import EngineGuidanceConfig, predict_engine_guided_moves
from persona_chess.evaluation.benchmark import run_benchmark
from persona_chess.evaluation.metrics import evaluate_move_matching
from persona_chess.exceptions import OptionalDependencyError
from persona_chess.model_card import build_model_card
from persona_chess.models.registry import supported_model_types
from persona_chess.neural import (
    AdapterManifest,
    LoraConfig,
    MoveVocabulary,
    NeuralAutoConfig,
    NeuralConfigProfile,
    NeuralTrainingConfig,
    PolicyBatch,
    PositionVocabulary,
    TransformerPolicyConfig,
    build_policy_samples,
    collate_policy_samples,
    create_adapter_manifest,
    create_adapter_manifest_from_vocabulary_sizes,
    iter_policy_batches,
    predict_policy_moves_from_checkpoint,
    prepare_streaming_neural_artifacts,
    recommend_neural_config,
    save_torch_policy_checkpoint,
    train_policy_model,
    train_policy_model_streaming,
    validate_neural_artifacts,
)
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.profile.builder import build_profile
from persona_chess.training import (
    build_training_records,
    iter_training_records,
    read_training_records_jsonl,
    write_training_records_jsonl,
)

DatasetOutputFormat = Literal["examples", "training"]
ModelCardOutputFormat = Literal["json", "markdown"]
T = TypeVar("T")

app = typer.Typer(
    help="Train and inspect lightweight chess personas from PGN files.",
    no_args_is_help=True,
)


@app.command()
def profile(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    out: Annotated[Path | None, typer.Option(help="Optional JSON output path.")] = None,
) -> None:
    persona_profile = build_profile(pgn, GameFilter(player=player, color=color))
    payload = persona_profile.to_dict()

    if out:
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"Wrote profile: {out}")
        return

    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("model-card")
def model_card(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    out: Annotated[Path | None, typer.Option(help="Optional report output path.")] = None,
    output_format: Annotated[
        ModelCardOutputFormat,
        typer.Option("--format", help="Report format."),
    ] = "json",
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    max_games: Annotated[
        int | None,
        typer.Option(help="Limit the number of matched games."),
    ] = None,
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when analyzing style."),
    ] = 0,
) -> None:
    card = build_model_card(
        pgn,
        player=player,
        color=color,
        max_games=max_games,
        skip_first_plies=skip_first_plies,
    )
    payload = (
        card.to_markdown()
        if output_format == "markdown"
        else json.dumps(card.to_dict(), indent=2, sort_keys=True)
    )

    if out:
        out.write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"Wrote model card: {out}")
        return

    typer.echo(payload)


@app.command()
def dataset(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    out: Annotated[Path, typer.Option(help="JSONL output path.")],
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when building examples."),
    ] = 0,
) -> None:
    examples = build_move_examples(
        pgn,
        GameFilter(player=player, color=color),
        skip_first_plies=skip_first_plies,
    )
    write_examples_jsonl(out, examples)
    typer.echo(f"Wrote {len(examples)} examples: {out}")


@app.command("export-training")
def export_training(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    out: Annotated[Path, typer.Option(help="Training JSONL output path.")],
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when exporting records."),
    ] = 0,
) -> None:
    examples = build_move_examples(
        pgn,
        GameFilter(player=player, color=color),
        skip_first_plies=skip_first_plies,
    )
    records = build_training_records(examples)
    write_training_records_jsonl(out, records)
    typer.echo(f"Wrote {len(records)} training records: {out}")


@app.command("export-training-stream")
def export_training_stream(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    out: Annotated[Path, typer.Option(help="Training JSONL output path.")],
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    max_games: Annotated[
        int | None,
        typer.Option(help="Limit the number of matched games."),
    ] = None,
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when exporting records."),
    ] = 0,
) -> None:
    examples = iter_move_examples(
        pgn,
        GameFilter(player=player, color=color, max_games=max_games),
        skip_first_plies=skip_first_plies,
    )
    written = write_training_records_jsonl(out, iter_training_records(examples))
    typer.echo(f"Wrote {written} training records: {out}")


@app.command("prepare-neural")
def prepare_neural(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    manifest_out: Annotated[Path, typer.Option(help="Adapter manifest output path.")],
    move_vocab_out: Annotated[
        Path,
        typer.Option("--move-vocab-out", "--vocab-out", help="Move vocabulary output path."),
    ],
    position_vocab_out: Annotated[
        Path,
        typer.Option(help="Position vocabulary output path."),
    ],
    base_model: Annotated[
        str,
        typer.Option(help="Base model identifier for the future adapter."),
    ] = "persona-chess/base-small",
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when preparing neural records."),
    ] = 0,
    config_profile: Annotated[
        NeuralConfigProfile,
        typer.Option(help="Hardware preset for omitted neural options."),
    ] = "auto",
    device: Annotated[
        str | None,
        typer.Option(help="Target Torch device for hardware-aware defaults, such as cpu or cuda."),
    ] = None,
    epochs: Annotated[
        int | None,
        typer.Option(help="Training epochs. Auto-selected when omitted."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Micro-batch size. Auto-selected when omitted."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option(help="Learning rate. Auto-selected when omitted."),
    ] = None,
    gradient_accumulation_steps: Annotated[
        int | None,
        typer.Option(help="Optimizer steps are run after this many batches."),
    ] = None,
    d_model: Annotated[
        int | None,
        typer.Option(help="Transformer hidden size. Auto-selected when omitted."),
    ] = None,
    n_layers: Annotated[
        int | None,
        typer.Option(help="Transformer layer count. Auto-selected when omitted."),
    ] = None,
    n_heads: Annotated[
        int | None,
        typer.Option(help="Transformer attention head count. Auto-selected when omitted."),
    ] = None,
    dropout: Annotated[
        float | None,
        typer.Option(help="Transformer dropout. Auto-selected when omitted."),
    ] = None,
    lora_rank: Annotated[
        int | None,
        typer.Option(help="LoRA rank. Auto-selected when omitted."),
    ] = None,
    lora_alpha: Annotated[
        int | None,
        typer.Option(help="LoRA alpha. Auto-selected when omitted."),
    ] = None,
    lora_dropout: Annotated[
        float | None,
        typer.Option(help="LoRA dropout. Auto-selected when omitted."),
    ] = None,
) -> None:
    examples = build_move_examples(
        pgn,
        GameFilter(player=player, color=color),
        skip_first_plies=skip_first_plies,
    )
    records = build_training_records(examples)
    auto_config = _resolve_neural_auto_config(
        len(records),
        config_profile=config_profile,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    move_vocabulary = MoveVocabulary.from_records(records)
    position_vocabulary = PositionVocabulary.from_records(records)
    manifest = create_adapter_manifest(
        records,
        player=player,
        base_model=base_model,
        transformer=auto_config.transformer,
        training=auto_config.training,
        lora=auto_config.lora,
    )

    move_vocabulary.save(move_vocab_out)
    position_vocabulary.save(position_vocab_out)
    manifest.save(manifest_out)
    typer.echo(_format_neural_config_summary(auto_config))
    typer.echo(f"Wrote neural manifest: {manifest_out}")
    typer.echo(f"Wrote move vocabulary: {move_vocab_out}")
    typer.echo(f"Wrote position vocabulary: {position_vocab_out}")


@app.command("prepare-neural-stream")
def prepare_neural_stream(
    training_records: Annotated[
        Path,
        typer.Argument(help="Training JSONL path from export-training-stream."),
    ],
    player: Annotated[str, typer.Argument(help="Player name for the adapter manifest.")],
    manifest_out: Annotated[Path, typer.Option(help="Adapter manifest output path.")],
    move_vocab_out: Annotated[
        Path,
        typer.Option("--move-vocab-out", "--vocab-out", help="Move vocabulary output path."),
    ],
    position_vocab_out: Annotated[Path, typer.Option(help="Position vocabulary output path.")],
    base_model: Annotated[
        str,
        typer.Option(help="Base model identifier for the future adapter."),
    ] = "persona-chess/base-small",
    config_profile: Annotated[
        NeuralConfigProfile,
        typer.Option(help="Hardware preset for omitted neural options."),
    ] = "auto",
    device: Annotated[
        str | None,
        typer.Option(help="Target Torch device for hardware-aware defaults, such as cpu or cuda."),
    ] = None,
    epochs: Annotated[
        int | None,
        typer.Option(help="Training epochs for the manifest. Auto-selected when omitted."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Batch size for streaming training. Auto-selected when omitted."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option(help="Learning rate. Auto-selected when omitted."),
    ] = None,
    gradient_accumulation_steps: Annotated[
        int | None,
        typer.Option(help="Optimizer steps are run after this many batches."),
    ] = None,
    d_model: Annotated[
        int | None,
        typer.Option(help="Transformer hidden size. Auto-selected when omitted."),
    ] = None,
    n_layers: Annotated[
        int | None,
        typer.Option(help="Transformer layer count. Auto-selected when omitted."),
    ] = None,
    n_heads: Annotated[
        int | None,
        typer.Option(help="Transformer attention head count. Auto-selected when omitted."),
    ] = None,
    dropout: Annotated[
        float | None,
        typer.Option(help="Transformer dropout. Auto-selected when omitted."),
    ] = None,
    lora_rank: Annotated[
        int | None,
        typer.Option(help="LoRA rank. Auto-selected when omitted."),
    ] = None,
    lora_alpha: Annotated[
        int | None,
        typer.Option(help="LoRA alpha. Auto-selected when omitted."),
    ] = None,
    lora_dropout: Annotated[
        float | None,
        typer.Option(help="LoRA dropout. Auto-selected when omitted."),
    ] = None,
) -> None:
    artifacts = prepare_streaming_neural_artifacts(read_training_records_jsonl(training_records))
    auto_config = _resolve_neural_auto_config(
        artifacts.training_examples,
        config_profile=config_profile,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    manifest = create_adapter_manifest_from_vocabulary_sizes(
        player=player,
        base_model=base_model,
        transformer=auto_config.transformer,
        training=auto_config.training,
        lora=auto_config.lora,
        move_vocabulary_size=artifacts.move_vocabulary.size,
        position_vocabulary_size=artifacts.position_vocabulary.size,
        training_examples=artifacts.training_examples,
    )

    artifacts.move_vocabulary.save(move_vocab_out)
    artifacts.position_vocabulary.save(position_vocab_out)
    manifest.save(manifest_out)
    typer.echo(_format_neural_config_summary(auto_config))
    typer.echo(f"Wrote neural manifest: {manifest_out}")
    typer.echo(f"Wrote move vocabulary: {move_vocab_out}")
    typer.echo(f"Wrote position vocabulary: {position_vocab_out}")
    typer.echo(f"Counted {artifacts.training_examples} streaming training records.")


@app.command("recommend-neural-config")
def recommend_neural_config_command(
    training_examples: Annotated[
        int | None,
        typer.Option(help="Number of training records to tune for."),
    ] = None,
    training_records: Annotated[
        Path | None,
        typer.Option(help="Training JSONL path to count and tune for."),
    ] = None,
    config_profile: Annotated[
        NeuralConfigProfile,
        typer.Option(help="Hardware preset to use for the recommendation."),
    ] = "auto",
    device: Annotated[
        str | None,
        typer.Option(help="Target Torch device for hardware-aware defaults, such as cpu or cuda."),
    ] = None,
) -> None:
    examples = _resolve_training_example_count(
        training_examples=training_examples,
        training_records=training_records,
    )
    auto_config = recommend_neural_config(
        examples,
        profile=config_profile,
        device=device,
    )
    typer.echo(json.dumps(auto_config.to_dict(), indent=2, sort_keys=True))


@app.command("validate-neural")
def validate_neural(
    manifest: Annotated[Path, typer.Argument(help="Adapter manifest path.")],
    move_vocab: Annotated[Path, typer.Argument(help="Move vocabulary path.")],
    position_vocab: Annotated[Path, typer.Argument(help="Position vocabulary path.")],
) -> None:
    validation = validate_neural_artifacts(
        manifest=AdapterManifest.load(manifest),
        move_vocabulary=MoveVocabulary.load(move_vocab),
        position_vocabulary=PositionVocabulary.load(position_vocab),
    )
    typer.echo(json.dumps(validation.to_dict(), indent=2, sort_keys=True))
    if not validation.ok:
        raise typer.Exit(code=1)


@app.command("train-neural")
def train_neural(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    checkpoint_dir: Annotated[Path, typer.Option(help="Checkpoint output directory.")],
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    max_games: Annotated[
        int | None,
        typer.Option(help="Limit the number of matched games."),
    ] = None,
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when preparing records."),
    ] = 0,
    use_lora: Annotated[
        bool,
        typer.Option("--use-lora/--full-finetune", help="Train a PEFT LoRA adapter."),
    ] = True,
    device: Annotated[str | None, typer.Option(help="Torch device, such as cpu or cuda.")] = None,
    config_profile: Annotated[
        NeuralConfigProfile,
        typer.Option(help="Hardware preset for omitted neural options."),
    ] = "auto",
    epochs: Annotated[
        int | None,
        typer.Option(help="Training epochs. Auto-selected when omitted."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Batch size. Auto-selected when omitted."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option(help="Learning rate. Auto-selected when omitted."),
    ] = None,
    gradient_accumulation_steps: Annotated[
        int | None,
        typer.Option(help="Optimizer steps are run after this many batches."),
    ] = None,
    d_model: Annotated[
        int | None,
        typer.Option(help="Transformer hidden size. Auto-selected when omitted."),
    ] = None,
    n_layers: Annotated[
        int | None,
        typer.Option(help="Transformer layer count. Auto-selected when omitted."),
    ] = None,
    n_heads: Annotated[
        int | None,
        typer.Option(help="Transformer attention head count. Auto-selected when omitted."),
    ] = None,
    dropout: Annotated[
        float | None,
        typer.Option(help="Transformer dropout. Auto-selected when omitted."),
    ] = None,
    lora_rank: Annotated[
        int | None,
        typer.Option(help="LoRA rank. Auto-selected when omitted."),
    ] = None,
    lora_alpha: Annotated[
        int | None,
        typer.Option(help="LoRA alpha. Auto-selected when omitted."),
    ] = None,
    lora_dropout: Annotated[
        float | None,
        typer.Option(help="LoRA dropout. Auto-selected when omitted."),
    ] = None,
) -> None:
    examples = build_move_examples(
        pgn,
        GameFilter(player=player, color=color, max_games=max_games),
        skip_first_plies=skip_first_plies,
    )
    records = build_training_records(examples)

    auto_config = _resolve_neural_auto_config(
        len(records),
        config_profile=config_profile,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    transformer = auto_config.transformer
    training = auto_config.training
    lora = auto_config.lora
    typer.echo(_format_neural_config_summary(auto_config))
    move_vocabulary = MoveVocabulary.from_records(records)
    position_vocabulary = PositionVocabulary.from_records(records)
    manifest = create_adapter_manifest(
        records,
        player=player,
        transformer=transformer,
        lora=lora,
        training=training,
    )
    samples = build_policy_samples(
        records,
        position_vocabulary=position_vocabulary,
        move_vocabulary=move_vocabulary,
        max_sequence_length=transformer.max_sequence_length,
    )
    batches = [
        collate_policy_samples(batch, move_pad_id=move_vocabulary.pad_id)
        for batch in _chunks(samples, training.batch_size)
    ]

    try:
        model, result = train_policy_model(
            batches,
            transformer=transformer,
            training=training,
            position_vocabulary_size=position_vocabulary.size,
            move_vocabulary_size=move_vocabulary.size,
            device=device,
            lora=lora if use_lora else None,
        )
        checkpoint = save_torch_policy_checkpoint(
            checkpoint_dir,
            model=model,
            adapter_manifest=manifest,
            move_vocabulary=move_vocabulary,
            position_vocabulary=position_vocabulary,
            training_result=result,
            lora_applied=use_lora,
        )
    except OptionalDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote neural checkpoint: {checkpoint_dir / checkpoint.model_state_file}")
    typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))


@app.command("train-neural-stream")
def train_neural_stream(
    training_records: Annotated[
        Path,
        typer.Argument(help="Training JSONL path from export-training-stream."),
    ],
    checkpoint_dir: Annotated[Path, typer.Option(help="Checkpoint output directory.")],
    manifest: Annotated[Path, typer.Option(help="Adapter manifest path.")],
    move_vocab: Annotated[Path, typer.Option(help="Move vocabulary path.")],
    position_vocab: Annotated[Path, typer.Option(help="Position vocabulary path.")],
    use_lora: Annotated[
        bool,
        typer.Option("--use-lora/--full-finetune", help="Train a PEFT LoRA adapter."),
    ] = True,
    device: Annotated[str | None, typer.Option(help="Torch device, such as cpu or cuda.")] = None,
    epochs: Annotated[
        int | None,
        typer.Option(help="Override training epochs from the manifest."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Override batch size from the manifest."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option(help="Override learning rate from the manifest."),
    ] = None,
    gradient_accumulation_steps: Annotated[
        int | None,
        typer.Option(help="Override gradient accumulation from the manifest."),
    ] = None,
    d_model: Annotated[
        int | None,
        typer.Option(help="Override transformer hidden size from the manifest."),
    ] = None,
    n_layers: Annotated[
        int | None,
        typer.Option(help="Override transformer layer count from the manifest."),
    ] = None,
    n_heads: Annotated[
        int | None,
        typer.Option(help="Override transformer attention head count from the manifest."),
    ] = None,
    dropout: Annotated[
        float | None,
        typer.Option(help="Override transformer dropout from the manifest."),
    ] = None,
    lora_rank: Annotated[
        int | None,
        typer.Option(help="Override LoRA rank from the manifest."),
    ] = None,
    lora_alpha: Annotated[
        int | None,
        typer.Option(help="Override LoRA alpha from the manifest."),
    ] = None,
    lora_dropout: Annotated[
        float | None,
        typer.Option(help="Override LoRA dropout from the manifest."),
    ] = None,
) -> None:
    adapter_manifest = AdapterManifest.load(manifest)
    move_vocabulary = MoveVocabulary.load(move_vocab)
    position_vocabulary = PositionVocabulary.load(position_vocab)
    validation = validate_neural_artifacts(
        manifest=adapter_manifest,
        move_vocabulary=move_vocabulary,
        position_vocabulary=position_vocabulary,
    )
    if not validation.ok:
        typer.echo(json.dumps(validation.to_dict(), indent=2, sort_keys=True), err=True)
        raise typer.Exit(code=1)
    adapter_manifest = _override_manifest_neural_config(
        adapter_manifest,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    typer.echo(_format_manifest_neural_config_summary(adapter_manifest))

    def batch_factory() -> Iterable[PolicyBatch]:
        return iter_policy_batches(
            read_training_records_jsonl(training_records),
            position_vocabulary=position_vocabulary,
            move_vocabulary=move_vocabulary,
            max_sequence_length=adapter_manifest.transformer.max_sequence_length,
            batch_size=adapter_manifest.training.batch_size,
        )

    try:
        model, result = train_policy_model_streaming(
            batch_factory,
            transformer=adapter_manifest.transformer,
            training=adapter_manifest.training,
            position_vocabulary_size=position_vocabulary.size,
            move_vocabulary_size=move_vocabulary.size,
            device=device,
            lora=adapter_manifest.lora if use_lora else None,
        )
        checkpoint = save_torch_policy_checkpoint(
            checkpoint_dir,
            model=model,
            adapter_manifest=adapter_manifest,
            move_vocabulary=move_vocabulary,
            position_vocabulary=position_vocabulary,
            training_result=result,
            lora_applied=use_lora,
        )
    except OptionalDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote neural checkpoint: {checkpoint_dir / checkpoint.model_state_file}")
    typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))


@app.command("neural-move")
def neural_move(
    checkpoint_dir: Annotated[Path, typer.Argument(help="Neural checkpoint directory.")],
    fen: Annotated[str, typer.Option(help="FEN string or 'startpos'.")],
    top_k: Annotated[int, typer.Option(help="Number of moves to return.")] = 3,
    device: Annotated[str | None, typer.Option(help="Torch device, such as cpu or cuda.")] = None,
) -> None:
    try:
        predictions = predict_policy_moves_from_checkpoint(
            checkpoint_dir,
            fen=fen,
            top_k=top_k,
            device=device,
        )
    except OptionalDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            [prediction.to_dict() for prediction in predictions],
            indent=2,
            sort_keys=True,
        )
    )


@app.command()
def split(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    train_out: Annotated[Path, typer.Option(help="Train split output path.")],
    test_out: Annotated[Path, typer.Option(help="Test split output path.")],
    validation_out: Annotated[
        Path | None, typer.Option(help="Optional validation output path.")
    ] = None,
    output_format: Annotated[
        DatasetOutputFormat,
        typer.Option(help="Output record type."),
    ] = "training",
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    test_ratio: Annotated[float, typer.Option(help="Held-out test ratio.")] = 0.2,
    validation_ratio: Annotated[float, typer.Option(help="Held-out validation ratio.")] = 0.0,
    seed: Annotated[int, typer.Option(help="Deterministic split seed.")] = 42,
) -> None:
    examples = build_move_examples(pgn, GameFilter(player=player, color=color))
    dataset_split = split_examples(
        examples,
        SplitConfig(test_ratio=test_ratio, validation_ratio=validation_ratio, seed=seed),
    )

    _write_split(train_out, dataset_split.train, output_format)
    _write_split(test_out, dataset_split.test, output_format)
    if validation_out:
        _write_split(validation_out, dataset_split.validation, output_format)

    typer.echo(json.dumps(dataset_split.to_summary(), indent=2, sort_keys=True))


@app.command()
def train(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    out: Annotated[Path, typer.Option(help="Persona artifact output path.")],
    model_type: Annotated[
        str,
        typer.Option(help=f"Model backend. Built-ins: {', '.join(supported_model_types())}."),
    ] = "blend",
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    max_games: Annotated[
        int | None,
        typer.Option(help="Limit the number of matched games."),
    ] = None,
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when fitting the persona."),
    ] = 0,
) -> None:
    persona = PersonaChess().fit_pgn(
        pgn,
        player=player,
        model_type=model_type,
        color=color,
        max_games=max_games,
        skip_first_plies=skip_first_plies,
    )
    persona.save(out)
    typer.echo(f"Wrote persona: {out}")


@app.command()
def move(
    model: Annotated[Path, typer.Argument(help="Path to a persona artifact.")],
    fen: Annotated[str, typer.Option(help="FEN string or 'startpos'.")],
    top_k: Annotated[int, typer.Option(help="Number of moves to return.")] = 3,
) -> None:
    persona = PersonaChess.load(model)
    predictions = [prediction.to_dict() for prediction in persona.predict(fen, top_k=top_k)]
    typer.echo(json.dumps(predictions, indent=2, sort_keys=True))


@app.command("engine-move")
def engine_move(
    model: Annotated[Path, typer.Argument(help="Path to a persona artifact.")],
    engine_path: Annotated[
        Path,
        typer.Option(help="Path to a UCI engine binary, such as Stockfish or Lc0."),
    ],
    fen: Annotated[str, typer.Option(help="FEN string or 'startpos'.")],
    top_k: Annotated[int, typer.Option(help="Number of moves to return.")] = 3,
    candidate_count: Annotated[
        int,
        typer.Option(help="Persona candidates to evaluate with the engine."),
    ] = 12,
    engine_weight: Annotated[
        float,
        typer.Option(help="Blend weight for engine quality in [0, 1]."),
    ] = 0.35,
    time_limit: Annotated[
        float | None,
        typer.Option(help="Engine analysis time per persona candidate, in seconds."),
    ] = 0.05,
    engine_depth: Annotated[
        int | None,
        typer.Option(help="Optional engine analysis depth per persona candidate."),
    ] = None,
) -> None:
    persona = PersonaChess.load(model)
    board = board_from_fen(fen)
    config = EngineGuidanceConfig(
        engine_weight=engine_weight,
        candidate_count=candidate_count,
        time_limit=time_limit,
        depth=engine_depth,
    )

    try:
        predictions = predict_engine_guided_moves(
            persona.require_model(),
            board=board,
            engine_path=engine_path,
            top_k=top_k,
            config=config,
        )
    except (OSError, ValueError, chess.engine.EngineError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps([prediction.to_dict() for prediction in predictions], indent=2))


@app.command()
def evaluate(
    model: Annotated[Path, typer.Argument(help="Path to a persona artifact.")],
    pgn: Annotated[Path, typer.Argument(help="Evaluation PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    k: Annotated[int, typer.Option(help="Top-k match threshold.")] = 3,
    max_games: Annotated[
        int | None,
        typer.Option(help="Limit the number of matched games."),
    ] = None,
    skip_first_plies: Annotated[
        int,
        typer.Option(help="Skip early plies when evaluating."),
    ] = 0,
) -> None:
    persona = PersonaChess.load(model)
    examples = build_move_examples(
        pgn,
        GameFilter(player=player, color=color, max_games=max_games),
        skip_first_plies=skip_first_plies,
    )
    metrics = evaluate_move_matching(persona.require_model(), examples, k=k)
    typer.echo(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))


@app.command()
def benchmark(
    pgn: Annotated[Path, typer.Argument(help="Path to a PGN file.")],
    player: Annotated[str, typer.Argument(help="Player name as written in PGN headers.")],
    model_type: Annotated[
        str,
        typer.Option(help=f"Model backend. Built-ins: {', '.join(supported_model_types())}."),
    ] = "blend",
    color: Annotated[PlayerColor, typer.Option(help="Filter games by player color.")] = "both",
    test_ratio: Annotated[float, typer.Option(help="Held-out test ratio.")] = 0.2,
    validation_ratio: Annotated[float, typer.Option(help="Held-out validation ratio.")] = 0.0,
    seed: Annotated[int, typer.Option(help="Deterministic split seed.")] = 42,
    k: Annotated[int, typer.Option(help="Top-k match threshold.")] = 3,
    out: Annotated[Path | None, typer.Option(help="Optional benchmark report path.")] = None,
) -> None:
    report = run_benchmark(
        pgn,
        player=player,
        model_type=model_type,
        color=color,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
        k=k,
    )
    payload = report.to_dict()
    if out:
        report.save(out)
        typer.echo(f"Wrote benchmark report: {out}")
        return

    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_neural_auto_config(
    training_examples: int,
    *,
    config_profile: NeuralConfigProfile,
    device: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    gradient_accumulation_steps: int | None,
    d_model: int | None,
    n_layers: int | None,
    n_heads: int | None,
    dropout: float | None,
    lora_rank: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
) -> NeuralAutoConfig:
    auto_config = recommend_neural_config(
        training_examples,
        profile=config_profile,
        device=device,
    )
    transformer = TransformerPolicyConfig(
        max_sequence_length=auto_config.transformer.max_sequence_length,
        d_model=d_model if d_model is not None else auto_config.transformer.d_model,
        n_layers=n_layers if n_layers is not None else auto_config.transformer.n_layers,
        n_heads=n_heads if n_heads is not None else auto_config.transformer.n_heads,
        dropout=dropout if dropout is not None else auto_config.transformer.dropout,
    )
    training = NeuralTrainingConfig(
        epochs=epochs if epochs is not None else auto_config.training.epochs,
        batch_size=batch_size if batch_size is not None else auto_config.training.batch_size,
        learning_rate=(
            learning_rate if learning_rate is not None else auto_config.training.learning_rate
        ),
        weight_decay=auto_config.training.weight_decay,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else auto_config.training.gradient_accumulation_steps
        ),
        warmup_ratio=auto_config.training.warmup_ratio,
        seed=auto_config.training.seed,
    )
    lora = LoraConfig(
        rank=lora_rank if lora_rank is not None else auto_config.lora.rank,
        alpha=lora_alpha if lora_alpha is not None else auto_config.lora.alpha,
        dropout=lora_dropout if lora_dropout is not None else auto_config.lora.dropout,
        target_modules=auto_config.lora.target_modules,
    )
    overrides = _neural_override_names(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    notes = (f"Manual overrides applied: {', '.join(overrides)}.",) if overrides else ()
    return auto_config.with_configs(
        transformer=transformer,
        training=training,
        lora=lora,
        notes=notes,
    )


def _override_manifest_neural_config(
    manifest: AdapterManifest,
    *,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    gradient_accumulation_steps: int | None,
    d_model: int | None,
    n_layers: int | None,
    n_heads: int | None,
    dropout: float | None,
    lora_rank: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
) -> AdapterManifest:
    overrides = _neural_override_names(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    if not overrides:
        return manifest

    transformer = TransformerPolicyConfig(
        max_sequence_length=manifest.transformer.max_sequence_length,
        d_model=d_model if d_model is not None else manifest.transformer.d_model,
        n_layers=n_layers if n_layers is not None else manifest.transformer.n_layers,
        n_heads=n_heads if n_heads is not None else manifest.transformer.n_heads,
        dropout=dropout if dropout is not None else manifest.transformer.dropout,
    )
    training = NeuralTrainingConfig(
        epochs=epochs if epochs is not None else manifest.training.epochs,
        batch_size=batch_size if batch_size is not None else manifest.training.batch_size,
        learning_rate=learning_rate
        if learning_rate is not None
        else manifest.training.learning_rate,
        weight_decay=manifest.training.weight_decay,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else manifest.training.gradient_accumulation_steps
        ),
        warmup_ratio=manifest.training.warmup_ratio,
        seed=manifest.training.seed,
    )
    lora = LoraConfig(
        rank=lora_rank if lora_rank is not None else manifest.lora.rank,
        alpha=lora_alpha if lora_alpha is not None else manifest.lora.alpha,
        dropout=lora_dropout if lora_dropout is not None else manifest.lora.dropout,
        target_modules=manifest.lora.target_modules,
    )
    return replace(manifest, transformer=transformer, training=training, lora=lora)


def _neural_override_names(
    *,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    gradient_accumulation_steps: int | None,
    d_model: int | None,
    n_layers: int | None,
    n_heads: int | None,
    dropout: float | None,
    lora_rank: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
) -> list[str]:
    values: list[tuple[str, object | None]] = [
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("learning_rate", learning_rate),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("d_model", d_model),
        ("n_layers", n_layers),
        ("n_heads", n_heads),
        ("dropout", dropout),
        ("lora_rank", lora_rank),
        ("lora_alpha", lora_alpha),
        ("lora_dropout", lora_dropout),
    ]
    return [name for name, value in values if value is not None]


def _format_neural_config_summary(auto_config: NeuralAutoConfig) -> str:
    transformer = auto_config.transformer
    training = auto_config.training
    lora = auto_config.lora
    return (
        "Selected neural config: "
        f"profile={auto_config.profile}, "
        f"device={auto_config.hardware.device_type}, "
        f"epochs={training.epochs}, "
        f"batch_size={training.batch_size}, "
        f"grad_accum={training.gradient_accumulation_steps}, "
        f"effective_batch={auto_config.effective_batch_size}, "
        f"d_model={transformer.d_model}, "
        f"layers={transformer.n_layers}, "
        f"heads={transformer.n_heads}, "
        f"lora_rank={lora.rank}"
    )


def _format_manifest_neural_config_summary(manifest: AdapterManifest) -> str:
    transformer = manifest.transformer
    training = manifest.training
    return (
        "Using neural manifest config: "
        f"epochs={training.epochs}, "
        f"batch_size={training.batch_size}, "
        f"grad_accum={training.gradient_accumulation_steps}, "
        f"effective_batch={training.batch_size * training.gradient_accumulation_steps}, "
        f"d_model={transformer.d_model}, "
        f"layers={transformer.n_layers}, "
        f"heads={transformer.n_heads}, "
        f"lora_rank={manifest.lora.rank}"
    )


def _resolve_training_example_count(
    *,
    training_examples: int | None,
    training_records: Path | None,
) -> int:
    if training_examples is not None:
        return _require_positive_training_examples(training_examples)
    if training_records is not None:
        return _require_positive_training_examples(
            sum(1 for _ in read_training_records_jsonl(training_records))
        )
    raise typer.BadParameter("Provide --training-examples or --training-records.")


def _require_positive_training_examples(training_examples: int) -> int:
    if training_examples <= 0:
        raise typer.BadParameter("training examples must be positive")
    return training_examples


def _write_split(
    path: Path, examples: list[MoveExample], output_format: DatasetOutputFormat
) -> None:
    if output_format == "examples":
        write_examples_jsonl(path, examples)
        return

    records = build_training_records(examples)
    write_training_records_jsonl(path, records)


def _chunks(items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


if __name__ == "__main__":
    app()
