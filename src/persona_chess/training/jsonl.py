import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from persona_chess.training.records import TrainingRecord


def write_training_records_jsonl(path: str | Path, records: Iterable[TrainingRecord]) -> int:
    output_path = Path(path)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


def read_training_records_jsonl(path: str | Path) -> Iterator[TrainingRecord]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if stripped := line.strip():
                yield TrainingRecord.from_dict(json.loads(stripped))


def count_training_records_jsonl(path: str | Path) -> int:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
