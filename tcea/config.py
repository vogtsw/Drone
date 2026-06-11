"""Central configuration for the TCEA pipeline (spec sections 7, 9, 10)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TceaConfig:
    # --- Monitor (spec 6.1 / 7: CUSUM drift detection) ---
    warmup_windows: int = 6          # windows used to learn the per-zone baseline
    cusum_k: float = 0.5             # CUSUM slack (in baseline std units)
    cusum_h: float = 5.0             # CUSUM decision threshold
    confirm_windows: int = 2         # consecutive significant windows before raising an event
    std_floor_db: float = 0.15       # floor for baseline std to avoid division blow-up

    # --- Measurement pairing ---
    samples_per_window: int = 80     # paired prediction/measurement samples per window per zone
    recent_batch_windows: int = 3    # windows of paired data kept for diagnosis tools

    # --- Agent budgets (spec GR4) ---
    diagnoser_tool_budget: int = 15
    planner_tool_budget: int = 8
    attribution_confidence_threshold: float = 0.7

    # --- Resource budgets (spec GR5) ---
    gpu_hours_budget: float = 24.0
    collection_flights_budget: int = 2

    # --- Validator gate (spec GR1) ---
    validation_rmse_db: float = 3.0      # frozen test-set RMSE must come back below this
    regression_delta_db: float = 0.75    # non-drift zones may not degrade by more than this
    max_plan_retries: int = 2

    # --- Frozen regression set / calibration ---
    frozen_points_per_zone: int = 120
    calibration_samples: int = 200

    # --- Confidence broadcast levels (spec G5) ---
    confidence_healthy: float = 0.9
    confidence_drifting: float = 0.5
    confidence_escalated: float = 0.3


DEFAULT_CONFIG = TceaConfig()
