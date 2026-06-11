"""P0 Publisher: canary release, automatic rollback, confidence broadcast.

Implements spec 6.1 step 6 and guardrail GR7: the staged model is committed,
re-checked on a fresh canary evaluation, and rolled back automatically if
the canary degrades. Zone confidence levels are broadcast to downstream
consumers (L1/L2) after every transition (spec G5).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from tcea.config import TceaConfig
from tcea.twin.simulator import DigitalTwinWorld, ZoneState


class Publisher:
    def __init__(self, world: DigitalTwinWorld, config: TceaConfig | None = None):
        self.world = world
        self.config = config or TceaConfig()
        self.confidence: dict[str, float] = {
            zone_id: self.config.confidence_healthy for zone_id in world.twin
        }
        self.broadcasts: list[dict[str, Any]] = []

    def broadcast_confidence(self, zone_id: str, level: float, reason: str) -> None:
        self.confidence[zone_id] = level
        self.broadcasts.append(
            {"zone": zone_id, "confidence": level, "reason": reason}
        )

    def _canary_rmse(self, zone_id: str) -> float:
        points = self.world.frozen_points[zone_id]
        measurements = self.world.measure(zone_id, points)
        predictions = self.world.predict(zone_id, points)
        return float(np.sqrt(np.mean((predictions - measurements) ** 2)))

    def canary_and_publish(self, staged: ZoneState) -> bool:
        """Commit staged → live, verify on a fresh canary sample, roll back
        automatically on failure. Returns True when fully published."""
        zone_id = staged.zone_id
        backup = copy.deepcopy(self.world.twin[zone_id])
        self.world.commit_zone(staged)

        canary_rmse = self._canary_rmse(zone_id)
        if canary_rmse <= self.config.validation_rmse_db:
            self.broadcast_confidence(
                zone_id, self.config.confidence_healthy,
                f"model published, canary rmse {canary_rmse:.2f} dB",
            )
            return True

        # GR7: automatic rollback on canary anomaly.
        self.world.commit_zone(backup)
        self.broadcast_confidence(
            zone_id, self.config.confidence_escalated,
            f"canary failed ({canary_rmse:.2f} dB), rolled back",
        )
        return False
