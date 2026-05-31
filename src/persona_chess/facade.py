import re
from datetime import datetime
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
from persona_chess.neural.hf_base import DEFAULT_BASE_MODEL
from persona_chess.neural.inference import predict_policy_moves_from_checkpoint
from persona_chess.neural.session import (
    NeuralTrainRequest,
    NeuralTrainResult,
    train_neural_persona,
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
        checkpoint_dir: str | Path | None = None,
        output_dir: str | Path = "checkpoints",
        base_model: str = DEFAULT_BASE_MODEL,
        init_checkpoint: str | Path | None = None,
        color: PlayerColor = "both",
        max_games: int | None = None,
        skip_first_plies: int = 0,
        use_lora: bool = True,
        device: str | None = None,
        config_profile: NeuralConfigProfile = "auto",
        validation_ratio: float = 0.1,
        streaming: bool | None = None,
        show_progress: bool = True,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        mixed_precision: MixedPrecisionMode | None = None,
    ) -> NeuralTrainResult:
        resolved_checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else _default_checkpoint_dir(output_dir, player=player)
        )
        result = train_neural_persona(
            NeuralTrainRequest(
                pgn=path,
                player=player,
                checkpoint_dir=resolved_checkpoint_dir,
                base_model=base_model,
                init_checkpoint=init_checkpoint,
                color=color,
                max_games=max_games,
                skip_first_plies=skip_first_plies,
                use_lora=use_lora,
                device=device,
                config_profile=config_profile,
                validation_ratio=validation_ratio,
                streaming=_should_stream(path) if streaming is None else streaming,
                show_progress=show_progress,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                mixed_precision=mixed_precision,
            )
        )
        self.neural_checkpoint_dir = result.checkpoint_dir
        return result

    def predict(self, fen: str, *, top_k: int = 1) -> list[MovePrediction]:
        return self.require_model().predict(board_from_fen(fen), top_k=top_k)

    def predict_neural(
        self,
        fen: str,
        *,
        checkpoint_dir: str | Path | None = None,
        top_k: int = 1,
        device: str | None = None,
        use_base_model: bool = True,
        base_model_weight: float = 0.65,
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
            use_base_model=use_base_model,
            base_model_weight=base_model_weight,
        )

    def move(
        self,
        fen: str,
        *,
        device: str | None = None,
        use_base_model: bool = True,
        base_model_weight: float = 0.65,
    ) -> MovePrediction:
        if self.neural_checkpoint_dir is not None:
            return self.predict_neural(
                fen,
                top_k=1,
                device=device,
                use_base_model=use_base_model,
                base_model_weight=base_model_weight,
            )[0]
        return self.predict(fen, top_k=1)[0]

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

    @classmethod
    def load_neural(cls, checkpoint_dir: str | Path) -> "PersonaChess":
        persona = cls()
        persona.neural_checkpoint_dir = Path(checkpoint_dir)
        return persona


def _default_checkpoint_dir(output_dir: str | Path, *, player: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path(output_dir) / f"{_slugify(player)}-{timestamp}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "persona"


def _should_stream(path: str | Path) -> bool:
    input_path = Path(path)
    try:
        return input_path.is_file() and input_path.stat().st_size >= 256 * 1024 * 1024
    except OSError:
        return False
