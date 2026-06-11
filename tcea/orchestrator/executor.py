"""Repair executor: deterministic action runner over a *staged* zone copy.

Nothing here touches the live twin (spec GR1/GR2): the orchestrator stages
a copy, the executor mutates it, and only the Publisher may commit it after
the Validator gate passes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tcea.models import RepairPlan
from tcea.tools.registry import ToolContext
from tcea.twin.simulator import ZoneState


class RepairExecutor:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def execute(self, plan: RepairPlan, staged: ZoneState) -> list[dict[str, Any]]:
        log: list[dict[str, Any]] = []
        for action in plan.actions:
            handler = getattr(self, f"_do_{action.name}", None)
            if handler is None:
                raise ValueError(f"unknown repair action: {action.name}")
            result = handler(staged, **action.params)
            log.append({"action": action.name, "result": result})
        return log

    # ------------------------------------------------------------------ #
    def _do_update_twin_config(self, staged: ZoneState,
                               changes: list[dict[str, Any]]) -> dict[str, Any]:
        applied = 0
        for change in changes:
            for gnb in staged.gnbs:
                if gnb.gnb_id == change["gnb_id"]:
                    setattr(gnb, change["field"], change["new"])
                    applied += 1
        return {"applied_changes": applied}

    def _do_update_twin_environment(self, staged: ZoneState,
                                    delta_att_db: float) -> dict[str, Any]:
        staged.env_attenuation_db += float(delta_att_db)
        return {"new_env_attenuation_db": round(staged.env_attenuation_db, 2)}

    def _do_regenerate_sim_data(self, staged: ZoneState) -> dict[str, Any]:
        # In the toy twin, predictions are computed on the fly from the zone
        # state, so "regenerating the simulated dataset" is a bookkeeping
        # step. It is kept as an explicit action to mirror the production
        # pipeline (targeted Sionna RT re-runs, spec 6.1 step 4).
        return {"regenerated_for_zone": staged.zone_id}

    def _do_calibrate_zone(self, staged: ZoneState) -> dict[str, Any]:
        """Incremental calibration: fit a residual bias offset on fresh
        measurements against the staged model (LoRA-offset analogue)."""
        world = self.ctx.world
        points = world._sample_points(staged.zone_id, self.ctx.config.calibration_samples)
        measurements = world.measure(staged.zone_id, points)
        predictions = world.predict(staged.zone_id, points, zone_override=staged)
        offset = float(np.mean(measurements - predictions))
        staged.bias_correction_db += offset
        return {"learned_offset_db": round(offset, 3),
                "total_bias_correction_db": round(staged.bias_correction_db, 3)}
