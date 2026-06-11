"""Benchmark smoke test: small run must meet the spec 10.1 acceptance bars."""

from tcea.benchmark.scenarios import run_benchmark


def test_benchmark_smoke():
    benchmark = run_benchmark(per_type=2, seed=42)
    summary = benchmark.summary()
    assert summary["scenarios"] == 6
    # spec 10.1: detection recall >= 0.9, attribution top-1 >= 0.75
    assert summary["detection_recall"] >= 0.9
    assert summary["attribution_top1_accuracy"] >= 0.75
    # repairable scenarios must close autonomously and effectively
    assert summary["repair_success_rate"] >= 0.75
    assert summary["autonomous_closure_rate"] >= 0.6
