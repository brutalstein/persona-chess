from pathlib import Path

from persona_chess.evaluation import BenchmarkReport, run_benchmark

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"


def test_run_benchmark_returns_versioned_split_report(tmp_path: Path) -> None:
    report = run_benchmark(
        FIXTURE,
        player="Target Player",
        model_type="blend",
        test_ratio=0.5,
        k=3,
    )

    assert report.schema_version == "persona-chess/benchmark-report/v1"
    assert report.model_type == "blend"
    assert [evaluation.name for evaluation in report.evaluations] == ["train", "test"]
    assert report.split["train_examples"] == 5
    assert report.split["test_examples"] == 5

    output = tmp_path / "benchmark.json"
    report.save(output)
    loaded = BenchmarkReport.load(output)

    assert loaded.to_dict() == report.to_dict()
