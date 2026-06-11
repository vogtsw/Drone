"""Top-level wiring: world + monitor + agent + orchestrator (spec 4.2 diagram)."""

from __future__ import annotations

from tcea.agents.policy import AgentPolicy, RuleBasedPolicy
from tcea.config import TceaConfig
from tcea.memory.case_store import CaseStore
from tcea.models import CaseReport, DriftEvent
from tcea.monitor.drift_monitor import DriftMonitor
from tcea.orchestrator.orchestrator import TceaOrchestrator
from tcea.tools.registry import ToolContext
from tcea.twin.simulator import DigitalTwinWorld


class TceaSystem:
    """One deployable TCEA instance bound to a digital-twin world."""

    def __init__(self, n_zones: int = 4, seed: int = 0,
                 config: TceaConfig | None = None,
                 policy: AgentPolicy | None = None):
        self.config = config or TceaConfig()
        self.world = DigitalTwinWorld(n_zones=n_zones, seed=seed, config=self.config)
        self.monitor = DriftMonitor(self.config)
        self.memory = CaseStore()
        ctx = ToolContext(
            world=self.world, monitor=self.monitor,
            memory=self.memory, config=self.config,
        )
        self.orchestrator = TceaOrchestrator(
            ctx, policy or RuleBasedPolicy(), self.config
        )
        self.window = 0

    # ------------------------------------------------------------------ #
    def run_window(self) -> list[DriftEvent]:
        """Advance one monitoring window across all zones; return new events."""
        events: list[DriftEvent] = []
        for zone_id in self.world.twin:
            batch = self.world.sample_window(zone_id, self.window)
            event = self.monitor.update(batch)
            if event is not None:
                events.append(event)
        self.window += 1
        return events

    def warm_up(self, extra_windows: int = 2) -> None:
        for _ in range(self.config.warmup_windows + extra_windows):
            self.run_window()

    def handle_event(self, event: DriftEvent) -> CaseReport:
        return self.orchestrator.handle_event(event)

    @property
    def confidence_map(self) -> dict[str, float]:
        return dict(self.orchestrator.publisher.confidence)
