"""Guardrail tests: GR1 validator gate, GR4 tool budget, GR5 cost budget."""

import dataclasses

import pytest

from tcea.config import TceaConfig
from tcea.models import CaseState, RootCause
from tcea.system import TceaSystem


def _trigger_event(system: TceaSystem, zone: str):
    system.warm_up()
    system.world.inject_environment_change(zone, delta_att_db=7.0, window=system.window)
    for _ in range(10):
        for event in system.run_window():
            if event.zone_id == zone:
                return event
    raise AssertionError("no event")


def test_gr4_tool_budget_exhaustion_escalates():
    config = dataclasses.replace(TceaConfig(), diagnoser_tool_budget=2)
    system = TceaSystem(n_zones=2, seed=21, config=config)
    event = _trigger_event(system, "Z-0001")
    report = system.handle_event(event)
    assert report.final_state == CaseState.ESCALATED
    assert "budget" in report.escalation_reason
    assert not report.published


def test_gr5_cost_budget_escalates():
    config = dataclasses.replace(TceaConfig(), gpu_hours_budget=1.0)
    system = TceaSystem(n_zones=2, seed=22, config=config)
    event = _trigger_event(system, "Z-0001")
    report = system.handle_event(event)
    assert report.final_state == CaseState.ESCALATED
    assert "exceeds budget" in report.escalation_reason


def test_gr1_validator_blocks_ineffective_repair():
    """If the executor is sabotaged into a no-op, the gate must hold and the
    live twin must remain untouched."""
    system = TceaSystem(n_zones=2, seed=23)
    zone = "Z-0001"
    orig = system.orchestrator.executor.execute
    system.orchestrator.executor.execute = lambda plan, staged: [{"action": "noop"}]

    event = _trigger_event(system, zone)
    twin_before = system.world.twin[zone].env_attenuation_db
    report = system.handle_event(event)

    assert report.final_state == CaseState.ESCALATED
    assert "validation failed" in report.escalation_reason
    assert not report.published
    assert system.world.twin[zone].env_attenuation_db == twin_before
    system.orchestrator.executor.execute = orig


def test_low_confidence_attribution_escalates():
    config = dataclasses.replace(TceaConfig(), attribution_confidence_threshold=0.99)
    system = TceaSystem(n_zones=2, seed=24, config=config)
    event = _trigger_event(system, "Z-0001")
    report = system.handle_event(event)
    assert report.final_state == CaseState.ESCALATED
    assert "confidence" in report.escalation_reason


def test_diagnosis_report_schema():
    system = TceaSystem(n_zones=2, seed=25)
    event = _trigger_event(system, "Z-0001")
    report = system.handle_event(event)
    payload = report.diagnosis.to_json()
    assert payload["zone"] == "Z-0001"
    assert payload["root_cause"]["category"] in [c.value for c in RootCause]
    assert 0.0 <= payload["root_cause"]["confidence"] <= 1.0
    assert payload["root_cause"]["evidence"], "evidence chain must not be empty"
