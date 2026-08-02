"""Feature flags for gradual rollout of architecture components.

Only flags with live production readers belong here (they are surfaced through
the ``feature_flags`` settings group in :mod:`victor.config.settings`). The
predictive-tools rollout cluster (master switch, rollout percentage, component
toggles, confidence threshold) was removed once call-graph audit F-019a
confirmed it had no production readers — the live predictive path in
``victor.agent.planning.tool_selection`` is constructor-gated, not
settings-gated.
"""

from __future__ import annotations

from pydantic import BaseModel


class FeatureFlagSettings(BaseModel):
    """Architecture feature flags still guarding live behavior."""

    use_composition_over_inheritance: bool = False
    use_strategy_based_tool_registration: bool = False
