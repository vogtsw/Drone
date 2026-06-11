"""End-to-end closed-loop tests for the two MVP root causes (spec 6.1)."""

import numpy as np
import pytest

from tcea.models import CaseState, RootCause
from tcea.system import TceaSystem


def _run_until_event(system: TceaSystem, zone_id: str, max_windows: int = 10):
    for _ in range(max_windows):
        for event in system.run_window():
            if event.zone_id == zone_id:
                return event
    raise AssertionError("drift was never detected")


def _post_repair_rmse(system: TceaSystem, zone_id: str) -> float:
    batch = system.world.sample_window(zone_id, window=999)
    return batch.rmse


@pytest.fixture
def system() -> TceaSystem:
    sys_ = TceaSystem(n_zones=3, seed=11)
    sys_.warm_up()
    return sys_


def test_config_change_loop(system: TceaSystem):
    zone = "Z-0001"
    system.world.inject_config_change(zone, 0, new_tilt_deg=9.5, window=system.window)
    event = _run_until_event(system, zone)
    report = system.handle_event(event)

    assert report.final_state == CaseState.DONE
    assert report.diagnosis.root_cause.category == RootCause.NETWORK_CONFIG_CHANGE
    assert report.diagnosis.root_cause.confidence >= 0.7
    assert report.published
    assert report.validation.passed
    # the twin must now carry the corrected tilt
    assert system.world.twin[zone].gnbs[0].tilt_deg == pytest.approx(9.5)
    # post-repair accuracy back at the noise floor
    assert _post_repair_rmse(system, zone) < system.config.validation_rmse_db
    # confidence restored after publication (spec G5)
    assert system.confidence_map[zone] == system.config.confidence_healthy
    # full audit trail recorded (spec GR8)
    states = [entry.get("state") for entry in report.audit_log if "state" in entry]
    assert states[0] == "DIAGNOSING" and states[-1] == "DONE"


def test_environment_change_loop(system: TceaSystem):
    zone = "Z-0002"
    system.world.inject_environment_change(zone, delta_att_db=7.0, window=system.window)
    event = _run_until_event(system, zone)
    report = system.handle_event(event)

    assert report.final_state == CaseState.DONE
    assert report.diagnosis.root_cause.category == RootCause.ENVIRONMENT_CHANGE
    assert report.published
    # twin env attenuation (incl. bias correction) compensates the 7 dB delta
    twin_zone = system.world.twin[zone]
    total_correction = (
        twin_zone.env_attenuation_db - 3.0 - twin_zone.bias_correction_db
    )
    assert total_correction == pytest.approx(7.0, abs=1.0)
    assert _post_repair_rmse(system, zone) < system.config.validation_rmse_db


def test_non_drift_zones_untouched(system: TceaSystem):
    zone = "Z-0000"
    other = "Z-0001"
    before = system.world.twin[other]
    before_tilt = before.gnbs[0].tilt_deg
    system.world.inject_environment_change(zone, delta_att_db=6.0, window=system.window)
    event = _run_until_event(system, zone)
    report = system.handle_event(event)
    assert report.final_state == CaseState.DONE
    after = system.world.twin[other]
    assert after.gnbs[0].tilt_deg == before_tilt
    assert after.bias_correction_db == 0.0


def test_data_quality_is_attributed_and_escalated(system: TceaSystem):
    zone = "Z-0000"
    system.world.inject_data_quality_issue(zone, noise_sigma=6.5, gps_ok_fraction=0.5)
    event = _run_until_event(system, zone)
    report = system.handle_event(event)

    # MVP boundary: correctly attributed, then handed to a human (spec 13)
    assert report.diagnosis.root_cause.category == RootCause.DATA_QUALITY
    assert report.final_state == CaseState.ESCALATED
    assert not report.published
    assert system.confidence_map[zone] == system.config.confidence_escalated


def test_case_memory_grows_after_done(system: TceaSystem):
    zone = "Z-0001"
    assert len(system.memory) == 0
    system.world.inject_environment_change(zone, delta_att_db=6.0, window=system.window)
    event = _run_until_event(system, zone)
    report = system.handle_event(event)
    assert report.final_state == CaseState.DONE
    assert len(system.memory) == 1
    record = system.memory.query(np.array([1.0, 0.0, 0.5, 0.0, 1.0]), top_k=1)[0][0]
    assert record.root_cause == RootCause.ENVIRONMENT_CHANGE
