import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from persona_chess.dataset.records import MoveExample


def write_examples_jsonl(path: str | Path, examples: Iterable[MoveExample]) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def read_examples_jsonl(path: str | Path) -> Iterator[MoveExample]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if stripped := line.strip():
                yield MoveExample.from_dict(json.loads(stripped))
