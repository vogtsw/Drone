"""Agent policies for the Diagnoser/Planner nodes (spec 4.1).

The agent *skeleton* (budgeted tool loop, structured output validation) is
deterministic; only the per-step decision is delegated to a policy. The
default ``RuleBasedPolicy`` is the deterministic baseline mandated by the
spec's ablation plan (section 10.3 / risk table "备选方案 A"); an LLM-backed
policy can be plugged in behind the same interface.
"""

from __future__ import annotations

from typing import Any, Protocol

from tcea.memory.case_store import drift_features
from tcea.models import DriftEvent, RootCause


class AgentPolicy(Protocol):
    def diagnose_step(self, event: DriftEvent, observations: dict[str, Any]) -> dict[str, Any]:
        """Return ``{"action": "tool", "tool": ..., "args": {...}}`` or a
        final structured verdict ``{"action": "final", "root_cause": {...},
        "alternatives": [...]}``."""
        ...

    def plan(self, diagnosis_payload: dict[str, Any], retry_count: int) -> dict[str, Any]:
        """Return a structured repair plan payload (see Planner)."""
        ...


# --------------------------------------------------------------------- #
# Rule-based deterministic policy
# --------------------------------------------------------------------- #
class RuleBasedPolicy:
    """Heuristic evidence-gathering sequence mirroring spec workflow 6.1."""

    CONFIG_LOOKBACK_WINDOWS = 10

    def diagnose_step(self, event: DriftEvent, obs: dict[str, Any]) -> dict[str, Any]:
        zone = event.zone_id

        if "error_stats" not in obs:
            return {"action": "tool", "tool": "query_error_stats",
                    "args": {"zone_id": zone}, "store_as": "error_stats"}

        if "quality" not in obs:
            return {"action": "tool", "tool": "query_measurement_quality",
                    "args": {"zone_id": zone}, "store_as": "quality"}

        if not obs["quality"]["quality_ok"]:
            return self._final_data_quality(obs)

        since = max(0, event.window - self.CONFIG_LOOKBACK_WINDOWS)
        if "config_diff" not in obs:
            return {"action": "tool", "tool": "diff_network_config",
                    "args": {"zone_id": zone, "since_window": since},
                    "store_as": "config_diff"}

        if "geo_diff" not in obs:
            return {"action": "tool", "tool": "diff_geo_environment",
                    "args": {"zone_id": zone, "since_window": since},
                    "store_as": "geo_diff"}

        if "memory" not in obs:
            features = drift_features(
                obs["error_stats"],
                has_config_change=bool(obs["config_diff"]["changes"]),
                has_geo_event=bool(obs["geo_diff"]["events"]),
            )
            return {"action": "tool", "tool": "query_case_memory",
                    "args": {"features": features.tolist()}, "store_as": "memory"}

        # Probe each candidate hypothesis once (cheap verification experiments).
        probes = obs.setdefault("probes", {})
        for category, hypothesis in self._candidate_hypotheses(obs).items():
            if category not in probes:
                return {"action": "tool", "tool": "run_probe_simulation",
                        "args": {"zone_id": zone, "hypothesis": hypothesis},
                        "store_as": f"probes.{category}"}

        return self._final_verdict(obs)

    # ------------------------------------------------------------------ #
    def _candidate_hypotheses(self, obs: dict[str, Any]) -> dict[str, dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        changes = obs["config_diff"]["changes"]
        if changes:
            candidates["network_config_change"] = {
                "category": "network_config_change",
                "changes": changes,
            }
        # The environment-offset hypothesis is always testable: the recent
        # bias is the maximum-likelihood uniform attenuation delta.
        candidates["environment_change"] = {
            "category": "environment_change",
            "delta_att_db": obs["error_stats"]["bias_db"],
        }
        return candidates

    def _final_data_quality(self, obs: dict[str, Any]) -> dict[str, Any]:
        q = obs["quality"]
        return {
            "action": "final",
            "root_cause": {
                "category": RootCause.DATA_QUALITY.value,
                "confidence": 0.9,
                "params": {},
                "evidence": [{
                    "tool": "query_measurement_quality",
                    "finding": (
                        f"gps_ok_fraction={q['gps_ok_fraction']:.2f}, "
                        f"n_samples={q['n_samples']} — measurement feed degraded"
                    ),
                }],
            },
            "alternatives": [],
        }

    def _final_verdict(self, obs: dict[str, Any]) -> dict[str, Any]:
        hypotheses = []
        memory_matches = obs.get("memory", {}).get("matches", [])
        for category, hypothesis in self._candidate_hypotheses(obs).items():
            probe = obs["probes"][category]
            confidence = 0.45 + 0.45 * probe["fit_score"]
            evidence = [{
                "tool": "run_probe_simulation",
                "finding": (
                    f"{category}: probe rmse "
                    f"{probe['rmse_probe_db']:.2f} dB vs now "
                    f"{probe['rmse_now_db']:.2f} dB (fit={probe['fit_score']:.2f})"
                ),
            }]
            # Corroborating records boost confidence (spec 6.1 evidence chain).
            if category == "network_config_change":
                changes = obs["config_diff"]["changes"]
                confidence += 0.05
                evidence.append({
                    "tool": "diff_network_config",
                    "finding": f"{len(changes)} logged config change(s): "
                               + ", ".join(
                                   f"{c['gnb_id']}.{c['field']} {c['old']}→{c['new']}"
                                   for c in changes),
                })
            elif category == "environment_change" and obs["geo_diff"]["events"]:
                confidence += 0.05
                evidence.append({
                    "tool": "diff_geo_environment",
                    "finding": obs["geo_diff"]["events"][0]["description"],
                })
            for match in memory_matches:
                if match["root_cause"] == category and match["similarity"] > 0.9:
                    confidence += 0.03
                    evidence.append({
                        "tool": "query_case_memory",
                        "finding": f"similar past case {match['case_id']} "
                                   f"(sim={match['similarity']:.2f}) had same root cause",
                    })
                    break
            hypotheses.append({
                "category": category,
                "confidence": min(confidence, 0.95),
                "params": {k: v for k, v in hypothesis.items() if k != "category"},
                "evidence": evidence,
            })

        if not hypotheses:
            return {
                "action": "final",
                "root_cause": {"category": RootCause.UNKNOWN.value,
                               "confidence": 0.3, "params": {}, "evidence": []},
                "alternatives": [],
            }

        hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        return {
            "action": "final",
            "root_cause": hypotheses[0],
            "alternatives": hypotheses[1:],
        }

    # ------------------------------------------------------------------ #
    # Planner policy (spec 6.1 step 3): minimum-cost repair DAG
    # ------------------------------------------------------------------ #
    def plan(self, diagnosis_payload: dict[str, Any], retry_count: int) -> dict[str, Any]:
        category = diagnosis_payload["root_cause"]["category"]
        params = diagnosis_payload["root_cause"]["params"]

        if category == RootCause.NETWORK_CONFIG_CHANGE.value:
            actions = [
                {"name": "update_twin_config",
                 "params": {"changes": params["changes"]}, "est_gpu_hours": 0.5},
                {"name": "regenerate_sim_data", "params": {}, "est_gpu_hours": 4.0},
                {"name": "calibrate_zone", "params": {}, "est_gpu_hours": 2.0},
            ]
        elif category == RootCause.ENVIRONMENT_CHANGE.value:
            actions = [
                {"name": "update_twin_environment",
                 "params": {"delta_att_db": params["delta_att_db"]}, "est_gpu_hours": 0.5},
                {"name": "regenerate_sim_data", "params": {}, "est_gpu_hours": 4.0},
                {"name": "calibrate_zone", "params": {}, "est_gpu_hours": 2.0},
            ]
        else:
            # MVP boundary (spec section 13): only the two root causes above
            # are repaired autonomously; everything else goes to a human.
            return {"escalate": True,
                    "reason": f"root cause '{category}' outside MVP repair scope"}

        if retry_count > 0:
            # Validation failed previously: add another calibration pass.
            actions.append({"name": "calibrate_zone", "params": {},
                            "est_gpu_hours": 2.0})
        return {"escalate": False, "actions": actions}
