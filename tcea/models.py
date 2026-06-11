"""Shared data models for the TCEA pipeline (spec section 8 schemas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RootCause(str, Enum):
    NETWORK_CONFIG_CHANGE = "network_config_change"
    ENVIRONMENT_CHANGE = "environment_change"
    DATA_QUALITY = "data_quality"
    MODEL_DEGRADATION = "model_degradation"
    UNKNOWN = "unknown"


class CaseState(str, Enum):
    IDLE = "IDLE"
    DIAGNOSING = "DIAGNOSING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    PUBLISHING = "PUBLISHING"
    DONE = "DONE"
    ESCALATED = "ESCALATED"
    ABORTED = "ABORTED"


@dataclass
class DriftEvent:
    event_id: str
    zone_id: str
    window: int
    rmse_baseline_db: float
    rmse_now_db: float

    def severity(self) -> dict[str, float]:
        return {
            "rmse_before_db": round(self.rmse_baseline_db, 2),
            "rmse_now_db": round(self.rmse_now_db, 2),
        }


@dataclass
class Evidence:
    tool: str
    finding: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    category: RootCause
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    # parameters needed to act on the hypothesis (e.g. config changes, delta dB)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosisReport:
    drift_event_id: str
    zone_id: str
    root_cause: Hypothesis
    alternatives: list[Hypothesis] = field(default_factory=list)
    tool_calls_used: int = 0

    def to_json(self) -> dict[str, Any]:
        def hyp(h: Hypothesis) -> dict[str, Any]:
            return {
                "category": h.category.value,
                "confidence": round(h.confidence, 3),
                "params": dict(h.params),
                "evidence": [
                    {"tool": e.tool, "finding": e.finding} for e in h.evidence
                ],
            }

        return {
            "drift_event_id": self.drift_event_id,
            "zone": self.zone_id,
            "root_cause": hyp(self.root_cause),
            "alternatives": [hyp(h) for h in self.alternatives],
            "tool_calls_used": self.tool_calls_used,
        }


@dataclass
class RepairAction:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    est_gpu_hours: float = 0.0
    est_flights: int = 0


@dataclass
class RepairPlan:
    plan_id: str
    zone_id: str
    actions: list[RepairAction]
    escalate: bool = False
    escalate_reason: str = ""

    @property
    def total_gpu_hours(self) -> float:
        return sum(a.est_gpu_hours for a in self.actions)

    @property
    def total_flights(self) -> int:
        return sum(a.est_flights for a in self.actions)


@dataclass
class ValidationResult:
    passed: bool
    rmse_after_db: float
    regression_ok: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseReport:
    """End-to-end audit record for one closed loop (spec G6)."""

    event: DriftEvent
    final_state: CaseState
    diagnosis: DiagnosisReport | None = None
    plan: RepairPlan | None = None
    validation: ValidationResult | None = None
    published: bool = False
    escalation_reason: str = ""
    transitions: list[str] = field(default_factory=list)
    audit_log: list[dict[str, Any]] = field(default_factory=list)
