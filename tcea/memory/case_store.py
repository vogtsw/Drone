"""P2 Memory: case library with similarity retrieval (spec 4.2 Memory).

Each closed loop stores a feature vector of the drift signature together
with the confirmed root cause and outcome. The Diagnoser retrieves the
top-k most similar past cases as few-shot anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tcea.models import RootCause


@dataclass
class CaseRecord:
    case_id: str
    zone_id: str
    features: np.ndarray
    root_cause: RootCause
    plan_summary: str
    repair_successful: bool
    notes: str = ""


def drift_features(error_stats: dict, has_config_change: bool,
                   has_geo_event: bool) -> np.ndarray:
    """Signature of a drift event used for case similarity."""
    return np.array([
        np.sign(error_stats.get("bias_db", 0.0)),
        np.clip(error_stats.get("height_error_corr", 0.0), -1.0, 1.0),
        np.clip(
            (error_stats.get("rmse_db", 0.0) - error_stats.get("baseline_rmse_db", 0.0))
            / 10.0,
            0.0, 1.0,
        ),
        1.0 if has_config_change else 0.0,
        1.0 if has_geo_event else 0.0,
    ])


class CaseStore:
    def __init__(self) -> None:
        self._cases: list[CaseRecord] = []

    def add(self, record: CaseRecord) -> None:
        self._cases.append(record)

    def __len__(self) -> int:
        return len(self._cases)

    def query(self, features: np.ndarray, top_k: int = 3) -> list[tuple[CaseRecord, float]]:
        scored: list[tuple[CaseRecord, float]] = []
        for case in self._cases:
            denom = np.linalg.norm(features) * np.linalg.norm(case.features)
            sim = float(np.dot(features, case.features) / denom) if denom > 1e-9 else 0.0
            scored.append((case, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
