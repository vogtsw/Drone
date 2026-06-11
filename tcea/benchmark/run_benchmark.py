"""CLI: run the injected-drift benchmark and print the scorecard.

Usage:
    python -m tcea.benchmark.run_benchmark --per-type 5 --seed 1
"""

from __future__ import annotations

import argparse
import json

from tcea.benchmark.scenarios import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="TCEA injected-drift benchmark")
    parser.add_argument("--per-type", type=int, default=5,
                        help="scenarios per root-cause type")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true",
                        help="print per-scenario rows")
    args = parser.parse_args()

    benchmark = run_benchmark(per_type=args.per_type, seed=args.seed)

    if args.verbose:
        header = (f"{'scenario':<28} {'detected':<9} {'win':<4} "
                  f"{'attributed':<24} {'ok':<3} {'state':<10} {'rmse_after':<10}")
        print(header)
        print("-" * len(header))
        for r in benchmark.results:
            print(
                f"{r.scenario.scenario_id:<28} "
                f"{str(r.detected):<9} "
                f"{str(r.windows_to_detect or '-'):<4} "
                f"{(r.attributed_cause.value if r.attributed_cause else '-'):<24} "
                f"{('Y' if r.attribution_correct else 'N'):<3} "
                f"{r.final_state:<10} "
                f"{(f'{r.rmse_after_db:.2f}' if r.rmse_after_db is not None else '-'):<10}"
            )
        print()

    print(json.dumps(benchmark.summary(), indent=2))


if __name__ == "__main__":
    main()
