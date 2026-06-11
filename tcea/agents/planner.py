"""Planner node: maps a diagnosis to a minimum-cost repair plan (spec 6.1 step 3).

Resource budgets (GR5) are enforced here: a plan whose estimated cost
exceeds the configured budgets is converted into an escalation.
"""

from __future__ import annotations

import itertools
from typing import Any

from tcea.config import TceaConfig
from tcea.models import DiagnosisReport, RepairAction, RepairPlan

_plan_counter = itertools.count(1)


class Planner:
    def __init__(self, policy: Any, config: TceaConfig | None = None):
        self.policy = policy
        self.config = config or TceaConfig()

    def plan(self, diagnosis: DiagnosisReport, retry_count: int = 0) -> RepairPlan:
        payload = self.policy.plan(diagnosis.to_json(), retry_count)
        plan_id = f"PLAN-{next(_plan_counter):04d}"

        if payload.get("escalate"):
            return RepairPlan(
                plan_id=plan_id, zone_id=diagnosis.zone_id, actions=[],
                escalate=True, escalate_reason=payload.get("reason", "policy escalated"),
            )

        actions = [
            RepairAction(
                name=a["name"],
                params=dict(a.get("params", {})),
                est_gpu_hours=float(a.get("est_gpu_hours", 0.0)),
                est_flights=int(a.get("est_flights", 0)),
            )
            for a in payload["actions"]
        ]
        plan = RepairPlan(plan_id=plan_id, zone_id=diagnosis.zone_id, actions=actions)

        # GR5: hard resource budget gate.
        if (plan.total_gpu_hours > self.config.gpu_hours_budget
                or plan.total_flights > self.config.collection_flights_budget):
            return RepairPlan(
                plan_id=plan_id, zone_id=diagnosis.zone_id, actions=[],
                escalate=True,
                escalate_reason=(
                    f"plan cost exceeds budget "
                    f"(gpu={plan.total_gpu_hours}h/{self.config.gpu_hours_budget}h, "
                    f"flights={plan.total_flights}/{self.config.collection_flights_budget})"
                ),
            )
        return plan
