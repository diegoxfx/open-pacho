import json

from openpacho.cli import main


def test_benchmark_json(tmp_path, capsys):
    path = tmp_path / "bench.json"
    assert (
        main(
            [
                "benchmark",
                "--backend",
                "numpy",
                "--size",
                "4",
                "--repeat",
                "2",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    report = json.loads(path.read_text())
    assert report["all_valid"]
    assert len(report["cases"]) == 3
    for case in report["cases"]:
        assert len(case["resident_seconds"]) == 2
        assert case["resident_median_seconds"] > 0
    assert json.loads(capsys.readouterr().out) == report


def test_invalid_benchmark(capsys):
    assert main(["benchmark", "--backend", "numpy", "--size", "0"]) == 2
    assert "positive" in capsys.readouterr().err
