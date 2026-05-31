from pathlib import Path

from persona_chess.model_card import PersonaModelCard, build_model_card
from persona_chess.model_card.types import MODEL_CARD_SCHEMA

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_build_model_card_summarizes_style_and_quality(tmp_path: Path) -> None:
    card = build_model_card(FIXTURE, player="Target Player")

    assert card.schema_version == MODEL_CARD_SCHEMA
    assert card.player == "Target Player"
    assert card.data_quality.games == 2
    assert card.data_quality.examples == 10
    assert card.data_quality.confidence == "low"
    assert "opening" in card.move_breakdown.phase_distribution
    assert card.recommendation.recommended_model == "blend"
    assert card.recommendation.neural_readiness == "not_ready"

    output = tmp_path / "target.model-card.json"
    card.save(output)

    assert PersonaModelCard.load(output).to_dict() == card.to_dict()


def test_model_card_markdown_contains_human_readable_sections() -> None:
    card = build_model_card(FIXTURE, player="Target Player")
    markdown = card.to_markdown()

    assert "# Persona Model Card: Target Player" in markdown
    assert "## Data Quality" in markdown
    assert "## Style" in markdown
    assert "## Recommendation" in markdown
