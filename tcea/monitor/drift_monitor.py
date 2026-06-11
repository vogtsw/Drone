"""P0 Monitor: per-zone error tracking + CUSUM drift detection (spec 6.1 / 7).

The monitor consumes one paired prediction/measurement batch per zone per
window, learns a baseline RMSE during warm-up, and raises a ``DriftEvent``
after the CUSUM statistic stays above threshold for ``confirm_windows``
consecutive windows.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tcea.config import TceaConfig
from tcea.models import DriftEvent
from tcea.twin.simulator import PairedBatch

_event_counter = itertools.count(1)


@dataclass
class ZoneBaseline:
    mean: float = 0.0
    std: float = 0.0
    n: int = 0
    _values: list[float] = field(default_factory=list)

    def add(self, value: float, std_floor: float) -> None:
        self._values.append(value)
        self.n = len(self._values)
        self.mean = float(np.mean(self._values))
        self.std = max(float(np.std(self._values)), std_floor)


class _ZoneTracker:
    def __init__(self, config: TceaConfig):
        self.config = config
        self.baseline = ZoneBaseline()
        self.cusum = 0.0
        self.significant_streak = 0
        self.active_event = False
        self.last_rmse = 0.0
        self.recent_batches: deque[PairedBatch] = deque(
            maxlen=config.recent_batch_windows
        )

    def update(self, batch: PairedBatch) -> bool:
        """Returns True when a *new* drift event should be raised."""
        rmse = batch.rmse
        self.last_rmse = rmse
        self.recent_batches.append(batch)

        if self.baseline.n < self.config.warmup_windows:
            self.baseline.add(rmse, self.config.std_floor_db)
            return False

        z = (rmse - self.baseline.mean) / self.baseline.std
        self.cusum = max(0.0, self.cusum + z - self.config.cusum_k)

        if self.cusum > self.config.cusum_h:
            self.significant_streak += 1
        else:
            self.significant_streak = 0

        if self.significant_streak >= self.config.confirm_windows and not self.active_event:
            self.active_event = True
            return True
        return False

    def reset_after_repair(self) -> None:
        """Re-learn the baseline after a model release (spec 6.1 step 7)."""
        self.baseline = ZoneBaseline()
        self.cusum = 0.0
        self.significant_streak = 0
        self.active_event = False


class DriftMonitor:
    def __init__(self, config: TceaConfig | None = None):
        self.config = config or TceaConfig()
        self.trackers: dict[str, _ZoneTracker] = {}

    def _tracker(self, zone_id: str) -> _ZoneTracker:
        if zone_id not in self.trackers:
            self.trackers[zone_id] = _ZoneTracker(self.config)
        return self.trackers[zone_id]

    def update(self, batch: PairedBatch) -> DriftEvent | None:
        tracker = self._tracker(batch.zone_id)
        if tracker.update(batch):
            return DriftEvent(
                event_id=f"DE-{next(_event_counter):04d}",
                zone_id=batch.zone_id,
                window=batch.window,
                rmse_baseline_db=tracker.baseline.mean,
                rmse_now_db=tracker.last_rmse,
            )
        return None

    def recent_batches(self, zone_id: str) -> list[PairedBatch]:
        return list(self._tracker(zone_id).recent_batches)

    def baseline_rmse(self, zone_id: str) -> float:
        return self._tracker(zone_id).baseline.mean

    def reset_zone(self, zone_id: str) -> None:
        self._tracker(zone_id).reset_after_repair()
