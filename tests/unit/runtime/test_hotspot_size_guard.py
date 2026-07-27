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
    "victor/agent/orchestrator.py": 4702,
    "victor/agent/services/planning_runtime.py": 3518,
    "victor/agent/services/tool_service.py": 3079,
    "victor/agent/services/runtime_intelligence.py": 2864,
    "victor/framework/vertical_integration.py": 2631,
    # +2 lines: FEP-0018 verifier wiring at the AgenticLoop construction site.
    "victor/agent/services/turn_execution_runtime.py": 2390,
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
