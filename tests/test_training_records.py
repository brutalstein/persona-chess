from pathlib import Path

from persona_chess.dataset.builder import build_move_examples, iter_move_examples
from persona_chess.pgn.filters import GameFilter
from persona_chess.training import (
    build_training_records,
    iter_training_records,
    read_training_records_jsonl,
    write_training_records_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_build_training_records_include_legal_move_mask() -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)

    assert records
    assert records[0].target_move == "e2e4"
    assert records[0].legal_moves[records[0].target_index] == "e2e4"
    assert records[0].legal_move_count == len(records[0].legal_moves)
    assert "e2e4" in records[0].legal_moves
    assert records[0].side_to_move == "white"


def test_training_records_round_trip_jsonl(tmp_path: Path) -> None:
    examples = build_move_examples(FIXTURE, GameFilter(player="Target Player"))
    records = build_training_records(examples)
    output = tmp_path / "records.jsonl"

    written = write_training_records_jsonl(output, records)
    loaded = list(read_training_records_jsonl(output))

    assert written == len(records)
    assert loaded == records


def test_streaming_training_records_match_in_memory_records() -> None:
    game_filter = GameFilter(player="Target Player")
    in_memory = build_training_records(build_move_examples(FIXTURE, game_filter))
    streamed = list(iter_training_records(iter_move_examples(FIXTURE, game_filter)))

    assert streamed == in_memory
