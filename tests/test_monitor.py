"""Monitor unit tests: baseline learning + CUSUM drift detection."""

from tcea.config import TceaConfig
from tcea.system import TceaSystem
from tcea.twin.simulator import DigitalTwinWorld


def test_baseline_rmse_is_near_noise_floor():
    world = DigitalTwinWorld(n_zones=2, seed=42)
    batch = world.sample_window("Z-0000", window=0)
    # With a perfectly synced twin the only error source is the 2 dB noise.
    assert 1.5 < batch.rmse < 2.6
    assert abs(batch.bias) < 0.8


def test_no_false_alarm_without_drift():
    system = TceaSystem(n_zones=2, seed=7)
    events = []
    for _ in range(25):
        events.extend(system.run_window())
    assert events == []


def test_cusum_detects_injected_config_drift():
    system = TceaSystem(n_zones=2, seed=3)
    system.warm_up()
    system.world.inject_config_change(
        "Z-0001", gnb_index=0, new_tilt_deg=9.0, window=system.window
    )
    detected = None
    for _ in range(10):
        for event in system.run_window():
            if event.zone_id == "Z-0001":
                detected = event
        if detected:
            break
    assert detected is not None
    assert detected.rmse_now_db > detected.rmse_baseline_db
    config = TceaConfig()
    # detection should be fast (confirm_windows + small margin)
    assert detected.window - system.config.warmup_windows < 10
    assert config.confirm_windows <= 3


def test_monitor_resets_after_repair():
    system = TceaSystem(n_zones=2, seed=5)
    system.warm_up()
    tracker_before = system.monitor.baseline_rmse("Z-0000")
    assert tracker_before > 0
    system.monitor.reset_zone("Z-0000")
    assert system.monitor.baseline_rmse("Z-0000") == 0.0
