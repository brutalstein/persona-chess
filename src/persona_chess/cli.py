import json
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import chess.engine
import typer

from persona_chess import PersonaChess
from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset import SplitConfig, split_examples
from persona_chess.dataset.builder import build_move_examples
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
    NeuralTrainingConfig,
    PositionVocabulary,
    TransformerPolicyConfig,
    build_policy_samples,
    collate_policy_samples,
    create_adapter_manifest,
    predict_policy_moves_from_checkpoint,
    save_torch_policy_checkpoint,
    train_policy_model,
    validate_neural_artifacts,
)
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.profile.builder import build_profile
from persona_chess.training import build_training_records, write_training_records_jsonl

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
) -> None:
    examples = build_move_examples(
        pgn,
        GameFilter(player=player, color=color),
        skip_first_plies=skip_first_plies,
    )
    records = build_training_records(examples)
    move_vocabulary = MoveVocabulary.from_records(records)
    position_vocabulary = PositionVocabulary.from_records(records)
    manifest = create_adapter_manifest(
        records,
        player=player,
        base_model=base_model,
    )

    move_vocabulary.save(move_vocab_out)
    position_vocabulary.save(position_vocab_out)
    manifest.save(manifest_out)
    typer.echo(f"Wrote neural manifest: {manifest_out}")
    typer.echo(f"Wrote move vocabulary: {move_vocab_out}")
    typer.echo(f"Wrote position vocabulary: {position_vocab_out}")


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
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 3,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 64,
    learning_rate: Annotated[float, typer.Option(help="Learning rate.")] = 3e-4,
    d_model: Annotated[int, typer.Option(help="Transformer hidden size.")] = 256,
    n_layers: Annotated[int, typer.Option(help="Transformer layer count.")] = 6,
    n_heads: Annotated[int, typer.Option(help="Transformer attention head count.")] = 8,
    dropout: Annotated[float, typer.Option(help="Transformer dropout.")] = 0.1,
    lora_rank: Annotated[int, typer.Option(help="LoRA rank.")] = 8,
    lora_alpha: Annotated[int, typer.Option(help="LoRA alpha.")] = 16,
    lora_dropout: Annotated[float, typer.Option(help="LoRA dropout.")] = 0.05,
) -> None:
    examples = build_move_examples(
        pgn,
        GameFilter(player=player, color=color, max_games=max_games),
        skip_first_plies=skip_first_plies,
    )
    records = build_training_records(examples)

    transformer = TransformerPolicyConfig(
        max_sequence_length=256,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    )
    training = NeuralTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    lora = LoraConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
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
