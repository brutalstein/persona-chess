from collections.abc import Callable
from typing import Any

from persona_chess.exceptions import ArtifactError
from persona_chess.models.base import PersonaModel
from persona_chess.models.blend import BlendPersonaModel
from persona_chess.models.frequency import FrequencyPersonaModel
from persona_chess.models.opening import OpeningBookPersonaModel
from persona_chess.models.phase import PhasePersonaModel

ModelLoader = Callable[[dict[str, Any]], PersonaModel]
ModelFactory = Callable[[], PersonaModel]

_LOADERS: dict[str, ModelLoader] = {}
_FACTORIES: dict[str, ModelFactory] = {}


def register_model(model_type: str, loader: ModelLoader, factory: ModelFactory) -> None:
    if not model_type:
        raise ValueError("model_type must not be empty")
    _LOADERS[model_type] = loader
    _FACTORIES[model_type] = factory


def create_model(model_type: str) -> PersonaModel:
    try:
        factory = _FACTORIES[model_type]
    except KeyError as exc:
        raise ArtifactError(f"Unsupported model type: {model_type}") from exc
    return factory()


def load_model(model_type: str, payload: dict[str, Any]) -> PersonaModel:
    try:
        loader = _LOADERS[model_type]
    except KeyError as exc:
        raise ArtifactError(f"Unsupported model type: {model_type}") from exc
    return loader(payload)


def supported_model_types() -> tuple[str, ...]:
    return tuple(sorted(_LOADERS))


register_model(
    BlendPersonaModel.model_type,
    BlendPersonaModel.from_payload,
    BlendPersonaModel,
)
register_model(
    FrequencyPersonaModel.model_type,
    FrequencyPersonaModel.from_payload,
    FrequencyPersonaModel,
)
register_model(
    OpeningBookPersonaModel.model_type,
    OpeningBookPersonaModel.from_payload,
    OpeningBookPersonaModel,
)
register_model(
    PhasePersonaModel.model_type,
    PhasePersonaModel.from_payload,
    PhasePersonaModel,
)
