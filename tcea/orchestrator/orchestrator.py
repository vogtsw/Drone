"""TCEA Orchestrator: deterministic six-stage state machine (spec sections 4, 7).

DIAGNOSING → PLANNING → EXECUTING → VALIDATING → PUBLISHING → DONE,
with escalation paths and a bounded validation-failure retry loop back to
PLANNING. Every transition and tool effect is written to the audit log
(spec GR8) and successful loops are stored in case memory.
"""

from __future__ import annotations

import itertools
from typing import Any

from tcea.agents.diagnoser import Diagnoser
from tcea.agents.planner import Planner
from tcea.config import TceaConfig
from tcea.memory.case_store import CaseRecord, CaseStore, drift_features
from tcea.models import (
    CaseReport,
    CaseState,
    DiagnosisReport,
    DriftEvent,
    RootCause,
)
from tcea.orchestrator.executor import RepairExecutor
from tcea.tools.registry import ToolContext, ToolRegistry
from tcea.validation.publisher import Publisher
from tcea.validation.validator import Validator

_case_counter = itertools.count(1)


class TceaOrchestrator:
    def __init__(self, ctx: ToolContext, policy: Any,
                 config: TceaConfig | None = None):
        self.ctx = ctx
        self.config = config or TceaConfig()
        self.registry = ToolRegistry(ctx)
        self.diagnoser = Diagnoser(self.registry, policy, self.config)
        self.planner = Planner(policy, self.config)
        self.executor = RepairExecutor(ctx)
        self.validator = Validator(ctx.world, ctx.monitor, self.config)
        self.publisher = Publisher(ctx.world, self.config)
        self.memory: CaseStore = ctx.memory

    # ------------------------------------------------------------------ #
    def handle_event(self, event: DriftEvent) -> CaseReport:
        report = CaseReport(event=event, final_state=CaseState.IDLE)

        def transition(state: CaseState, **details: Any) -> None:
            report.final_state = state
            report.transitions.append(state.value)
            report.audit_log.append({"state": state.value, **details})

        self.publisher.broadcast_confidence(
            event.zone_id, self.config.confidence_drifting,
            f"drift event {event.event_id} raised",
        )

        # ---- DIAGNOSING -------------------------------------------------
        transition(CaseState.DIAGNOSING, event=event.severity())
        diagnosis = self.diagnoser.diagnose(event)
        if diagnosis is None:
            return self._escalate(report, "diagnoser tool budget exhausted (GR4)")
        report.diagnosis = diagnosis
        report.audit_log.append({"diagnosis": diagnosis.to_json()})

        if diagnosis.root_cause.confidence < self.config.attribution_confidence_threshold:
            return self._escalate(
                report,
                f"attribution confidence {diagnosis.root_cause.confidence:.2f} "
                f"< {self.config.attribution_confidence_threshold}",
            )

        # ---- PLANNING / EXECUTING / VALIDATING retry loop ---------------
        for retry in range(self.config.max_plan_retries + 1):
            transition(CaseState.PLANNING, retry=retry)
            plan = self.planner.plan(diagnosis, retry_count=retry)
            report.plan = plan
            if plan.escalate:
                return self._escalate(report, plan.escalate_reason)

            transition(CaseState.EXECUTING, plan_id=plan.plan_id,
                       actions=[a.name for a in plan.actions])
            staged = self.ctx.world.stage_zone(event.zone_id)
            exec_log = self.executor.execute(plan, staged)
            report.audit_log.append({"execution": exec_log})

            transition(CaseState.VALIDATING)
            validation = self.validator.validate(staged)
            report.validation = validation
            report.audit_log.append({
                "validation": {
                    "passed": validation.passed,
                    "rmse_after_db": round(validation.rmse_after_db, 2),
                    "regression_ok": validation.regression_ok,
                },
            })
            if validation.passed:
                break
            # staged copy is simply discarded; live twin untouched (GR1)
        else:
            return self._escalate(
                report,
                f"validation failed after {self.config.max_plan_retries + 1} attempts",
            )

        # ---- PUBLISHING --------------------------------------------------
        transition(CaseState.PUBLISHING)
        if not self.publisher.canary_and_publish(staged):
            return self._escalate(report, "canary failed, rolled back (GR7)")
        report.published = True

        # ---- DONE: memory write + monitor reset --------------------------
        self.ctx.monitor.reset_zone(event.zone_id)
        self._store_case(report, diagnosis)
        transition(CaseState.DONE)
        return report

    # ------------------------------------------------------------------ #
    def _escalate(self, report: CaseReport, reason: str) -> CaseReport:
        report.final_state = CaseState.ESCALATED
        report.escalation_reason = reason
        report.transitions.append(CaseState.ESCALATED.value)
        report.audit_log.append({"state": CaseState.ESCALATED.value, "reason": reason})
        self.publisher.broadcast_confidence(
            report.event.zone_id, self.config.confidence_escalated,
            f"escalated: {reason}",
        )
        return report

    def _store_case(self, report: CaseReport, diagnosis: DiagnosisReport) -> None:
        error_stats = next(
            (log.result for log in self.registry.call_log
             if log.tool == "query_error_stats"
             and log.args.get("zone_id") == report.event.zone_id),
            {},
        )
        has_config = diagnosis.root_cause.category == RootCause.NETWORK_CONFIG_CHANGE
        has_geo = any(
            e.tool == "diff_geo_environment" for e in diagnosis.root_cause.evidence
        )
        self.memory.add(CaseRecord(
            case_id=f"CASE-{next(_case_counter):04d}",
            zone_id=report.event.zone_id,
            features=drift_features(error_stats, has_config, has_geo),
            root_cause=diagnosis.root_cause.category,
            plan_summary=", ".join(a.name for a in report.plan.actions) if report.plan else "",
            repair_successful=report.published,
        ))
