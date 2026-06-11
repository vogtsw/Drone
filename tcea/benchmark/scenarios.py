"""P1 injected-drift benchmark (spec 10.3 评估方法 #1).

Each scenario builds a fresh world, injects a known root cause into a
random zone, and runs the full detect→diagnose→repair→publish loop. The
ground truth is known by construction, so detection recall, attribution
accuracy, repair effectiveness and autonomous-closure rate can be scored
exactly (spec 10.1 metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tcea.models import CaseState, RootCause
from tcea.system import TceaSystem

MAX_DETECTION_WINDOWS = 10

# Root causes that the MVP repairs autonomously; others must be escalated
# with the *correct* attribution (spec section 13 MVP boundary).
AUTONOMOUS_CAUSES = {RootCause.NETWORK_CONFIG_CHANGE, RootCause.ENVIRONMENT_CHANGE}


@dataclass
class Scenario:
    scenario_id: str
    root_cause: RootCause
    seed: int
    magnitude: float  # tilt delta (deg) or attenuation delta (dB) or noise sigma


@dataclass
class ScenarioResult:
    scenario: Scenario
    detected: bool = False
    windows_to_detect: int | None = None
    attributed_cause: RootCause | None = None
    attribution_correct: bool = False
    final_state: str = ""
    published: bool = False
    rmse_after_db: float | None = None
    repair_effective: bool = False
    autonomous: bool = False
    expected_autonomous: bool = True


@dataclass
class BenchmarkResult:
    results: list[ScenarioResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = len(self.results)
        detected = [r for r in self.results if r.detected]
        repairable = [r for r in self.results if r.expected_autonomous and r.detected]
        det_windows = [r.windows_to_detect for r in detected
                       if r.windows_to_detect is not None]
        return {
            "scenarios": n,
            "detection_recall": round(len(detected) / n, 3) if n else 0.0,
            "mean_windows_to_detect": (
                round(float(np.mean(det_windows)), 2) if det_windows else None
            ),
            "attribution_top1_accuracy": (
                round(sum(r.attribution_correct for r in detected) / len(detected), 3)
                if detected else 0.0
            ),
            "repair_success_rate": (
                round(sum(r.repair_effective for r in repairable) / len(repairable), 3)
                if repairable else None
            ),
            "autonomous_closure_rate": (
                round(sum(r.autonomous for r in repairable) / len(repairable), 3)
                if repairable else None
            ),
        }


def _inject(system: TceaSystem, scenario: Scenario, zone_id: str) -> None:
    rng = np.random.default_rng(scenario.seed)
    if scenario.root_cause == RootCause.NETWORK_CONFIG_CHANGE:
        gnb_index = int(rng.integers(0, 2))
        old_tilt = system.world.truth[zone_id].gnbs[gnb_index].tilt_deg
        system.world.inject_config_change(
            zone_id, gnb_index, old_tilt + scenario.magnitude, window=system.window,
        )
    elif scenario.root_cause == RootCause.ENVIRONMENT_CHANGE:
        system.world.inject_environment_change(
            zone_id, scenario.magnitude, window=system.window,
        )
    elif scenario.root_cause == RootCause.DATA_QUALITY:
        system.world.inject_data_quality_issue(zone_id, noise_sigma=scenario.magnitude)
    else:
        raise ValueError(f"unsupported scenario cause: {scenario.root_cause}")


def run_scenario(scenario: Scenario, n_zones: int = 4) -> ScenarioResult:
    system = TceaSystem(n_zones=n_zones, seed=scenario.seed)
    result = ScenarioResult(
        scenario=scenario,
        expected_autonomous=scenario.root_cause in AUTONOMOUS_CAUSES,
    )

    system.warm_up()
    rng = np.random.default_rng(scenario.seed + 1)
    zone_id = f"Z-{int(rng.integers(0, n_zones)):04d}"
    inject_window = system.window
    _inject(system, scenario, zone_id)

    event = None
    for _ in range(MAX_DETECTION_WINDOWS):
        events = system.run_window()
        match = next((e for e in events if e.zone_id == zone_id), None)
        if match is not None:
            event = match
            break
    if event is None:
        return result

    result.detected = True
    result.windows_to_detect = event.window - inject_window + 1

    report = system.handle_event(event)
    result.final_state = report.final_state.value
    result.published = report.published
    if report.diagnosis is not None:
        result.attributed_cause = report.diagnosis.root_cause.category
        result.attribution_correct = (
            report.diagnosis.root_cause.category == scenario.root_cause
        )
    if report.validation is not None:
        result.rmse_after_db = report.validation.rmse_after_db
    result.autonomous = report.final_state == CaseState.DONE
    result.repair_effective = (
        report.published
        and report.validation is not None
        and report.validation.rmse_after_db <= system.config.validation_rmse_db
    )
    return result


def default_scenarios(per_type: int = 5, seed: int = 1) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    scenarios: list[Scenario] = []
    spec = [
        (RootCause.NETWORK_CONFIG_CHANGE, lambda: float(rng.uniform(2.5, 5.0))),
        (RootCause.ENVIRONMENT_CHANGE, lambda: float(rng.uniform(4.0, 9.0))),
        (RootCause.DATA_QUALITY, lambda: float(rng.uniform(5.0, 8.0))),
    ]
    for cause, magnitude_fn in spec:
        for i in range(per_type):
            scenarios.append(Scenario(
                scenario_id=f"{cause.value}-{i}",
                root_cause=cause,
                seed=int(rng.integers(0, 2**31 - 1)),
                magnitude=magnitude_fn(),
            ))
    return scenarios


def run_benchmark(per_type: int = 5, seed: int = 1) -> BenchmarkResult:
    result = BenchmarkResult()
    for scenario in default_scenarios(per_type=per_type, seed=seed):
        result.results.append(run_scenario(scenario))
    return result
