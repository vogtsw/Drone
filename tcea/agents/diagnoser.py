"""Diagnoser node: budgeted ReAct loop over the tool registry (spec 6.1 step 2).

The loop skeleton is deterministic and enforces the GR4 tool budget; the
per-step decision comes from the pluggable policy. The final verdict is
validated against a minimal schema (GR6) before being converted into a
``DiagnosisReport``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tcea.config import TceaConfig
from tcea.models import DiagnosisReport, DriftEvent, Evidence, Hypothesis, RootCause
from tcea.tools.registry import ToolRegistry


@dataclass
class DiagnoserState:
    event: DriftEvent
    observations: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0


def _validate_final_payload(payload: dict[str, Any]) -> None:
    root = payload.get("root_cause")
    if not isinstance(root, dict):
        raise ValueError("final payload missing 'root_cause' object")
    for key in ("category", "confidence", "params", "evidence"):
        if key not in root:
            raise ValueError(f"root_cause missing required key '{key}'")
    RootCause(root["category"])  # raises on unknown category
    if not 0.0 <= float(root["confidence"]) <= 1.0:
        raise ValueError("confidence must be within [0, 1]")


def _to_hypothesis(payload: dict[str, Any]) -> Hypothesis:
    return Hypothesis(
        category=RootCause(payload["category"]),
        confidence=float(payload["confidence"]),
        params=dict(payload.get("params", {})),
        evidence=[
            Evidence(tool=e["tool"], finding=e["finding"])
            for e in payload.get("evidence", [])
        ],
    )


class Diagnoser:
    def __init__(self, registry: ToolRegistry, policy: Any,
                 config: TceaConfig | None = None):
        self.registry = registry
        self.policy = policy
        self.config = config or TceaConfig()

    def diagnose(self, event: DriftEvent) -> DiagnosisReport | None:
        """Returns a report, or None when the tool budget is exhausted (→ escalate)."""
        state = DiagnoserState(event=event)

        while state.tool_calls < self.config.diagnoser_tool_budget:
            decision = self.policy.diagnose_step(event, state.observations)

            if decision["action"] == "final":
                _validate_final_payload(decision)
                return DiagnosisReport(
                    drift_event_id=event.event_id,
                    zone_id=event.zone_id,
                    root_cause=_to_hypothesis(decision["root_cause"]),
                    alternatives=[
                        _to_hypothesis(h) for h in decision.get("alternatives", [])
                    ],
                    tool_calls_used=state.tool_calls,
                )

            if decision["action"] != "tool":
                raise ValueError(f"policy returned unknown action: {decision['action']}")

            result = self.registry.call(decision["tool"], **decision["args"])
            state.tool_calls += 1
            store_as = decision.get("store_as", decision["tool"])
            if store_as.startswith("probes."):
                state.observations.setdefault("probes", {})[
                    store_as.split(".", 1)[1]
                ] = result
            else:
                state.observations[store_as] = result

        return None  # budget exhausted → orchestrator escalates (GR4)
