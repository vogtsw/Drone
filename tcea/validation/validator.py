"""P0 Validator gate (spec GR1): frozen test set + no-regression check.

A staged repair may only be published when
  1. the drift zone's RMSE on the frozen point set returns below threshold, and
  2. no other zone degrades beyond ``regression_delta_db`` versus its
     monitored baseline (catastrophic-forgetting guard).
"""

from __future__ import annotations

import numpy as np

from tcea.config import TceaConfig
from tcea.models import ValidationResult
from tcea.monitor.drift_monitor import DriftMonitor
from tcea.twin.simulator import DigitalTwinWorld, ZoneState


class Validator:
    def __init__(self, world: DigitalTwinWorld, monitor: DriftMonitor,
                 config: TceaConfig | None = None):
        self.world = world
        self.monitor = monitor
        self.config = config or TceaConfig()

    def _frozen_rmse(self, zone_id: str, zone_override: ZoneState | None = None) -> float:
        points = self.world.frozen_points[zone_id]
        measurements = self.world.measure(zone_id, points)
        predictions = self.world.predict(zone_id, points, zone_override=zone_override)
        return float(np.sqrt(np.mean((predictions - measurements) ** 2)))

    def validate(self, staged: ZoneState) -> ValidationResult:
        rmse_after = self._frozen_rmse(staged.zone_id, zone_override=staged)

        regression_ok = True
        regression_details: dict[str, float] = {}
        for zone_id in self.world.twin:
            if zone_id == staged.zone_id:
                continue
            baseline = self.monitor.baseline_rmse(zone_id)
            if baseline <= 0.0:
                continue  # zone has no learned baseline yet
            rmse = self._frozen_rmse(zone_id)
            regression_details[zone_id] = round(rmse - baseline, 3)
            if rmse - baseline > self.config.regression_delta_db:
                regression_ok = False

        passed = rmse_after <= self.config.validation_rmse_db and regression_ok
        return ValidationResult(
            passed=passed,
            rmse_after_db=rmse_after,
            regression_ok=regression_ok,
            details={
                "threshold_db": self.config.validation_rmse_db,
                "non_drift_zone_delta_db": regression_details,
            },
        )
