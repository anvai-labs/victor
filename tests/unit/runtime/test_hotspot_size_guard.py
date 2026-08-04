"""Guard test for hotspot file regrowth (TD-14, TD-15).

TD-R1 decomposed the orchestrator to 3,510 lines, then it silently regrew to
4,690 by 2026-07 because nothing ratcheted it. This guard pins the audited
sizes of the known hotspot files so they can only shrink: raising a cap
requires editing this file and explaining why in review.

As decomposition work lands (TD-14 orchestrator, TD-15 services sprawl),
lower the caps to the new audited sizes.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Audited 2026-07-02; ratcheted down 2026-07-25 (F2, foundations strategy §3.2).
# Caps may only be lowered, never raised.
HOTSPOT_LINE_CAPS = {
    # Calibrated to develop's size when the guard arrived via the main->develop
    # back-merge (the guard was born on main pinned to main's sizes and never saw
    # develop's pre-existing growth). The no-raise ratchet binds from HERE forward.
    # Ratcheted 2026-07-30 (ADR-019 increment 1): 4690 -> 4600 after extracting the
    # pure task-report metadata builders to victor/agent/task_report_metadata.py.
    # Ratcheted 2026-07-30 (ADR-019 increment 2): 4600 -> 4402 after extracting the
    # tool-supply policy (Tool-Necessity Gate + budgeter) to victor/agent/tool_supply_policy.py.
    # Ratcheted 2026-08-03 (ADR-019 increment 3): 4402 -> 4400 after moving
    # task-report start/finish metadata assembly into task_report_metadata.py.
    # Ratcheted 2026-08-03 (ADR-019 increment 4): 4400 -> 4368 after moving
    # the edge-model tool-necessity decision into tool_supply_policy.py.
    # Ratcheted 2026-08-03 (ADR-019 increment 5): 4368 -> 4340 after moving
    # KV tool ordering and strategy-setting interpretation into ToolStrategyRuntime.
    # Ratcheted 2026-08-03 (ADR-019 increment 6): 4340 -> 4326 after moving
    # provider-economics session locking into ToolStrategyRuntime.
    # Ratcheted 2026-08-03 (ADR-019 increment 7): 4326 -> 4320 after moving
    # strategy feature checks and ToolService utility delegation into ToolStrategyRuntime.
    # Ratcheted 2026-08-03 (ADR-019 increment 8): 4320 -> 4303 after moving
    # context-aware strategy execution and telemetry assembly into ToolStrategyRuntime.
    # Ratcheted 2026-08-03 (ADR-019 increment 9): 4303 -> 4290 after moving
    # tool-strategy metrics event emission into ToolStrategyRuntime.
    # Ratcheted 2026-08-03 (ADR-019 increment 10): 4290 -> 4266 after moving
    # configured KV strategy execution and session-cache updates into ToolStrategyRuntime.
    # Ratcheted 2026-08-03 (ADR-019 increment 11): 4266 -> 4225 after deleting
    # uncalled private KV session-lock and Gemini-provider compatibility helpers.
    "victor/agent/orchestrator.py": 4225,
    "victor/agent/services/planning_runtime.py": 3518,
    "victor/agent/services/tool_service.py": 3079,
    "victor/agent/services/runtime_intelligence.py": 2864,
    "victor/framework/vertical_integration.py": 2631,
    # FEP-0030 Phase 2: _build_rubric_complete_fn delegates to the judge-backend
    # resolver (judge_calibration_gate.build_judge_complete_fn) — shrinks the hotspot.
    "victor/agent/services/turn_execution_runtime.py": 2378,
    # F-004: package-ified tool_selection; parent capped at current size
    # (extraction deferred — this ratchet only prevents further growth).
    "victor/agent/tool_selection/selector.py": 2765,
    # F-003: context_compactor is a 1827-LOC god-object; decomposition deferred as
    # low-ROI (works, not a proven bottleneck). Ratchet prevents further growth.
    "victor/agent/context_compactor.py": 1827,
    # Audited 2026-07-22. [CANONICAL] chat service (1,740 LOC) — the largest
    # service after orchestrator; bind it so "chat operations" don't quietly
    # sprawl the way the orchestrator facade did.
    "victor/agent/services/chat_service.py": 1739,
}


class TestHotspotSizeGuard:
    """Prevent audited hotspot files from regrowing."""

    @pytest.mark.parametrize("rel_path,cap", sorted(HOTSPOT_LINE_CAPS.items()))
    def test_hotspot_does_not_regrow(self, rel_path: str, cap: int) -> None:
        path = REPO_ROOT / rel_path
        assert path.is_file(), (
            f"{rel_path} no longer exists — if it was decomposed or renamed, "
            f"remove or update its entry in HOTSPOT_LINE_CAPS."
        )
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        assert lines <= cap, (
            f"{rel_path} is {lines} lines (ratchet cap {cap}). "
            f"This file already regrew once after being decomposed (TD-R1 → TD-14); "
            f"move new behavior into the owning service or a new module instead of "
            f"growing the hotspot. If a cap must move, lower it — never raise it."
        )
