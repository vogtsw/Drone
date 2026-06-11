"""P1 tool layer (spec section 5): idempotent, read-only or sandboxed tools.

All tools are pure functions over the ``ToolContext``; nothing here mutates
the live twin — repair actions operate on a *staged* zone copy provided by
the orchestrator (spec GR1/GR2 boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tcea.config import TceaConfig
from tcea.memory.case_store import CaseStore
from tcea.monitor.drift_monitor import DriftMonitor
from tcea.twin.simulator import DigitalTwinWorld, ZoneState


@dataclass
class ToolContext:
    world: DigitalTwinWorld
    monitor: DriftMonitor
    memory: CaseStore
    config: TceaConfig


@dataclass
class ToolCallLog:
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]


class ToolRegistry:
    """Named tool dispatch with an audit trail of every call."""

    def __init__(self, context: ToolContext):
        self.ctx = context
        self.call_log: list[ToolCallLog] = []
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "query_error_stats": self.query_error_stats,
            "query_measurement_quality": self.query_measurement_quality,
            "diff_network_config": self.diff_network_config,
            "diff_geo_environment": self.diff_geo_environment,
            "query_case_memory": self.query_case_memory,
            "run_probe_simulation": self.run_probe_simulation,
        }

    def call(self, tool: str, **args: Any) -> dict[str, Any]:
        if tool not in self._tools:
            raise KeyError(f"unknown tool: {tool}")
        result = self._tools[tool](**args)
        self.call_log.append(ToolCallLog(tool=tool, args=args, result=result))
        return result

    # ------------------------------------------------------------------ #
    # Diagnostic tools
    # ------------------------------------------------------------------ #
    def query_error_stats(self, zone_id: str) -> dict[str, Any]:
        batches = self.ctx.monitor.recent_batches(zone_id)
        if not batches:
            return {"n_samples": 0}
        errors = np.concatenate([b.errors for b in batches])
        heights = np.concatenate([b.points[:, 2] for b in batches])
        corr = 0.0
        if np.std(errors) > 1e-9 and np.std(heights) > 1e-9:
            corr = float(np.corrcoef(errors, heights)[0, 1])
        return {
            "zone_id": zone_id,
            "n_samples": int(len(errors)),
            "rmse_db": float(np.sqrt(np.mean(errors**2))),
            "bias_db": float(np.mean(errors)),
            "height_error_corr": corr,
            "baseline_rmse_db": self.ctx.monitor.baseline_rmse(zone_id),
        }

    def query_measurement_quality(self, zone_id: str) -> dict[str, Any]:
        batches = self.ctx.monitor.recent_batches(zone_id)
        n = sum(len(b.points) for b in batches)
        gps_ok = self.ctx.world.gps_ok_fraction[zone_id]
        return {
            "zone_id": zone_id,
            "n_samples": n,
            "gps_ok_fraction": gps_ok,
            "quality_ok": bool(gps_ok >= 0.8 and n >= 50),
        }

    def diff_network_config(self, zone_id: str, since_window: int = 0) -> dict[str, Any]:
        changes = [
            {
                "gnb_id": c.gnb_id,
                "field": c.field_name,
                "old": c.old_value,
                "new": c.new_value,
                "window": c.window,
            }
            for c in self.ctx.world.config_log
            if c.zone_id == zone_id and c.window >= since_window
        ]
        return {"zone_id": zone_id, "changes": changes}

    def diff_geo_environment(self, zone_id: str, since_window: int = 0) -> dict[str, Any]:
        events = [
            {"description": e.description, "window": e.window}
            for e in self.ctx.world.geo_events
            if e.zone_id == zone_id and e.window >= since_window
        ]
        return {"zone_id": zone_id, "events": events}

    def query_case_memory(self, features: list[float], top_k: int = 3) -> dict[str, Any]:
        matches = self.ctx.memory.query(np.asarray(features, dtype=float), top_k=top_k)
        return {
            "matches": [
                {
                    "case_id": c.case_id,
                    "root_cause": c.root_cause.value,
                    "similarity": round(sim, 3),
                    "repair_successful": c.repair_successful,
                }
                for c, sim in matches
            ]
        }

    def run_probe_simulation(self, zone_id: str, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """Probe: 'if root cause X were true, would the twin error vanish?'

        Applies the hypothesised correction to a *scratch copy* of the twin
        zone and replays the recent paired windows. fit_score in [0, 1]
        measures how much of the excess error the hypothesis explains.
        """
        batches = self.ctx.monitor.recent_batches(zone_id)
        if not batches:
            return {"fit_score": 0.0, "rmse_probe_db": float("nan")}

        staged = self.ctx.world.stage_zone(zone_id)
        self._apply_hypothesis(staged, hypothesis)

        sq_errors: list[np.ndarray] = []
        for batch in batches:
            preds = self.ctx.world.predict(zone_id, batch.points, zone_override=staged)
            sq_errors.append((preds - batch.measurements) ** 2)
        rmse_probe = float(np.sqrt(np.mean(np.concatenate(sq_errors))))

        rmse_now = float(np.sqrt(np.mean(
            np.concatenate([b.errors for b in batches]) ** 2
        )))
        baseline = self.ctx.monitor.baseline_rmse(zone_id)
        excess = max(rmse_now - baseline, 1e-6)
        fit = float(np.clip((rmse_now - rmse_probe) / excess, 0.0, 1.0))
        return {
            "zone_id": zone_id,
            "rmse_now_db": rmse_now,
            "rmse_probe_db": rmse_probe,
            "fit_score": fit,
        }

    # ------------------------------------------------------------------ #
    # Hypothesis application (shared with the repair executor)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_hypothesis(staged: ZoneState, hypothesis: dict[str, Any]) -> None:
        category = hypothesis.get("category")
        if category == "network_config_change":
            for change in hypothesis.get("changes", []):
                for gnb in staged.gnbs:
                    if gnb.gnb_id == change["gnb_id"]:
                        setattr(gnb, change["field"], change["new"])
        elif category == "environment_change":
            staged.env_attenuation_db += float(hypothesis.get("delta_att_db", 0.0))
        else:
            raise ValueError(f"probe does not support category: {category}")
