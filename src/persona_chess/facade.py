from pathlib import Path

from persona_chess.chess.legal import board_from_fen
from persona_chess.dataset.builder import build_move_examples
from persona_chess.dataset.records import MoveExample
from persona_chess.exceptions import ModelNotFittedError
from persona_chess.models.base import PersonaModel
from persona_chess.models.registry import create_model, load_model
from persona_chess.models.types import MovePrediction
from persona_chess.pgn.filters import GameFilter, PlayerColor
from persona_chess.profile.builder import build_profile
from persona_chess.profile.types import PersonaProfile
from persona_chess.storage.artifact import PersonaArtifact


class PersonaChess:
    def __init__(self) -> None:
        self.profile: PersonaProfile | None = None
        self.model: PersonaModel | None = None
        self.examples: list[MoveExample] = []

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

    def predict(self, fen: str, *, top_k: int = 1) -> list[MovePrediction]:
        return self.require_model().predict(board_from_fen(fen), top_k=top_k)

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
