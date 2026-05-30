import pytest

from persona_chess.exceptions import ArtifactError
from persona_chess.models import (
    BlendPersonaModel,
    FrequencyPersonaModel,
    OpeningBookPersonaModel,
    PhasePersonaModel,
)
from persona_chess.models.registry import create_model, load_model, supported_model_types


def test_builtin_models_are_registered() -> None:
    assert supported_model_types() == (
        BlendPersonaModel.model_type,
        FrequencyPersonaModel.model_type,
        OpeningBookPersonaModel.model_type,
        PhasePersonaModel.model_type,
    )


def test_create_model_uses_registry() -> None:
    assert isinstance(create_model("blend"), BlendPersonaModel)


def test_load_model_rejects_unknown_type() -> None:
    with pytest.raises(ArtifactError):
        load_model("missing-model", {})
