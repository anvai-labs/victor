# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Classification backend selection (mirrors FEP-0012 ``DecisionBackend``).

A single ``classification_backend`` enum config replaces direct reads of the
``USE_TIERED_CLASSIFICATION`` feature flag for choosing whether classification
triage routes through the ``TieredDecisionService``. ``AUTO`` (default) preserves
prior behavior by consulting that flag; ``TIERED`` forces tiered classification.

Values:
- ``AUTO`` (default): use the TieredDecisionService iff
  ``USE_TIERED_CLASSIFICATION`` is enabled (legacy behavior).
- ``TIERED``: always use the TieredDecisionService for classification triage.

Aligned with the FEP-0012 "one knob" pattern.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ClassificationBackend(str, Enum):
    """Whether classification triage uses the ``TieredDecisionService``."""

    AUTO = "auto"
    TIERED = "tiered"

    @classmethod
    def parse(cls, value: Any) -> "ClassificationBackend":
        """Coerce a raw config value to a backend (falls back to AUTO)."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.AUTO


def use_tiered_classification() -> bool:
    """True when classification triage should use the ``TieredDecisionService``.

    Explicit ``TIERED`` wins. ``AUTO`` (default) consults the legacy
    ``USE_TIERED_CLASSIFICATION`` feature flag, preserving every existing config.
    """
    # Lazy import avoids a settings <-> backend module cycle.
    from victor.config.classification_settings import ClassificationSettings

    backend = ClassificationBackend.parse(ClassificationSettings().classification_backend)
    if backend is ClassificationBackend.TIERED:
        return True

    # AUTO -> legacy flag-based selection.
    try:
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        return is_feature_enabled(FeatureFlag.USE_TIERED_CLASSIFICATION)
    except Exception:
        return False
