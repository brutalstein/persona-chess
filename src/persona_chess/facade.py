from pathlib import Path

from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset.builder import build_move_examples
from persona_chess.dataset.records import MoveExample
from persona_chess.exceptions import ModelNotFittedError
from persona_chess.models.base import PersonaModel
from persona_chess.models.registry import create_model, load_model
from persona_chess.models.types import MovePrediction
from persona_chess.neural.autotune import NeuralConfigProfile
from persona_chess.neural.config import MixedPrecisionMode
from persona_chess.neural.inference import predict_policy_moves_from_checkpoint
from persona_chess.neural.session import (
    NeuralRecordsTrainRequest,
    NeuralTrainRequest,
    NeuralTrainResult,
    train_neural_persona,
    train_neural_records,
    write_pgn_training_records,
)
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.profile.builder import build_profile
from persona_chess.profile.types import PersonaProfile
from persona_chess.storage.artifact import PersonaArtifact


class PersonaChess:
    def __init__(self) -> None:
        self.profile: PersonaProfile | None = None
        self.model: PersonaModel | None = None
        self.examples: list[MoveExample] = []
        self.neural_checkpoint_dir: Path | None = None

    def fit_pgn(
        self,
        path: str | Path,
        *,
        player: str,
        model_type: str = "blend",
        color: PlayerColor = "both",
        max_games: int | None = None,
        skip_first_plies: int = 0,
    ) -> "PersonaChess":
        game_filter = GameFilter(player=player, color=color, max_games=max_games)
        examples = build_move_examples(path, game_filter, skip_first_plies=skip_first_plies)
        profile = build_profile(path, game_filter)
        return self.fit_examples(examples, profile=profile, model_type=model_type)

    def fit_examples(
        self,
        examples: list[MoveExample],
        *,
        profile: PersonaProfile,
        model_type: str = "blend",
    ) -> "PersonaChess":
        model = create_model(model_type)
        model.fit(examples)

        self.examples = examples
        self.profile = profile
        self.model = model
        return self

    def train(
        self,
        path: str | Path,
        *,
        player: str,
        checkpoint_dir: str | Path,
        base_model: str = "persona-chess/base-small",
        init_checkpoint: str | Path | None = None,
        resume_checkpoint: str | Path | None = None,
        model_registry: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
        color: PlayerColor = "both",
        max_games: int | None = None,
        skip_first_plies: int = 0,
        use_lora: bool = True,
        device: str | None = None,
        config_profile: NeuralConfigProfile = "auto",
        standard_position_vocabulary: bool = True,
        validation_ratio: float = 0.1,
        streaming: bool = False,
        records_dir: str | Path | None = None,
        save_best: bool = True,
        checkpoint_every_epoch: bool = False,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        warmup_ratio: float | None = None,
        max_grad_norm: float | None = None,
        mixed_precision: MixedPrecisionMode | None = None,
        gradient_accumulation_steps: int | None = None,
        d_model: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        dropout: float | None = None,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float | None = None,
    ) -> NeuralTrainResult:
        result = train_neural_persona(
            NeuralTrainRequest(
                pgn=path,
                player=player,
                checkpoint_dir=checkpoint_dir,
                base_model=base_model,
                init_checkpoint=init_checkpoint,
                resume_checkpoint=resume_checkpoint,
                model_registry=model_registry,
                model_cache_dir=model_cache_dir,
                color=color,
                max_games=max_games,
                skip_first_plies=skip_first_plies,
                use_lora=use_lora,
                device=device,
                config_profile=config_profile,
                standard_position_vocabulary=standard_position_vocabulary,
                validation_ratio=validation_ratio,
                streaming=streaming,
                records_dir=records_dir,
                save_best=save_best,
                checkpoint_every_epoch=checkpoint_every_epoch,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm,
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        )
        self.neural_checkpoint_dir = result.checkpoint_dir
        return result

    def train_records(
        self,
        training_records: str | Path,
        *,
        player: str,
        checkpoint_dir: str | Path,
        validation_records: str | Path | None = None,
        base_model: str = "persona-chess/base-small",
        init_checkpoint: str | Path | None = None,
        resume_checkpoint: str | Path | None = None,
        model_registry: str | Path | None = None,
        model_cache_dir: str | Path | None = None,
        use_lora: bool = True,
        device: str | None = None,
        config_profile: NeuralConfigProfile = "auto",
        standard_position_vocabulary: bool = True,
        save_best: bool = True,
        checkpoint_every_epoch: bool = False,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        warmup_ratio: float | None = None,
        max_grad_norm: float | None = None,
        mixed_precision: MixedPrecisionMode | None = None,
        gradient_accumulation_steps: int | None = None,
        d_model: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        dropout: float | None = None,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float | None = None,
    ) -> NeuralTrainResult:
        result = train_neural_records(
            NeuralRecordsTrainRequest(
                training_records=training_records,
                validation_records=validation_records,
                player=player,
                checkpoint_dir=checkpoint_dir,
                base_model=base_model,
                init_checkpoint=init_checkpoint,
                resume_checkpoint=resume_checkpoint,
                model_registry=model_registry,
                model_cache_dir=model_cache_dir,
                use_lora=use_lora,
                device=device,
                config_profile=config_profile,
                standard_position_vocabulary=standard_position_vocabulary,
                save_best=save_best,
                checkpoint_every_epoch=checkpoint_every_epoch,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm,
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        )
        self.neural_checkpoint_dir = result.checkpoint_dir
        return result

    def export_training_records(
        self,
        path: str | Path,
        out: str | Path,
        *,
        player: str,
        color: PlayerColor = "both",
        max_games: int | None = None,
        skip_first_plies: int = 0,
    ) -> int:
        return write_pgn_training_records(
            path,
            out,
            player=player,
            color=color,
            max_games=max_games,
            skip_first_plies=skip_first_plies,
        )

    def predict(self, fen: str, *, top_k: int = 1) -> list[MovePrediction]:
        return self.require_model().predict(board_from_fen(fen), top_k=top_k)

    def predict_neural(
        self,
        fen: str,
        *,
        checkpoint_dir: str | Path | None = None,
        top_k: int = 1,
        device: str | None = None,
    ) -> list[MovePrediction]:
        active_checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else self.neural_checkpoint_dir
        )
        if active_checkpoint_dir is None:
            raise ModelNotFittedError(
                "Train a neural persona or provide checkpoint_dir before requesting predictions."
            )
        return predict_policy_moves_from_checkpoint(
            active_checkpoint_dir,
            fen=fen,
            top_k=top_k,
            device=device,
        )

    def save(self, path: str | Path) -> None:
        if self.profile is None:
            raise ModelNotFittedError("Fit a persona before saving it.")

        model = self.require_model()
        artifact = PersonaArtifact.create(
            model_type=model.model_type,
            profile=self.profile,
            payload=model.to_payload(),
        )
        artifact.save(path)

    def require_model(self) -> PersonaModel:
        if self.model is None:
            raise ModelNotFittedError("Fit or load a persona before requesting predictions.")
        return self.model

    @classmethod
    def load(cls, path: str | Path) -> "PersonaChess":
        artifact = PersonaArtifact.load(path)

        persona = cls()
        persona.profile = artifact.profile
        persona.model = load_model(artifact.model_type, artifact.payload)
        return persona
