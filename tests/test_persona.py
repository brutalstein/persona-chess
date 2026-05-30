from pathlib import Path

from persona_chess import PersonaChess

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_fit_predict_save_load(tmp_path: Path) -> None:
    persona = PersonaChess().fit_pgn(FIXTURE, player="Target Player")

    predictions = persona.predict("startpos", top_k=2)
    assert predictions[0].move_uci == "e2e4"

    artifact = tmp_path / "target.persona.json"
    persona.save(artifact)

    loaded = PersonaChess.load(artifact)
    loaded_predictions = loaded.predict("startpos", top_k=1)

    assert loaded.profile is not None
    assert loaded.profile.player == "Target Player"
    assert loaded_predictions[0].move_uci == "e2e4"
