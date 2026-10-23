# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Decision-backend selection (FEP-0012).

A single ``decision_backend`` enum config replaces the prior tangle of boolean
feature flags (``USE_EDGE_MODEL`` / ``USE_LLM_DECISION_SERVICE`` / a would-be
``USE_LOCAL_CLASSIFIER``) and their implicit, fragile precedence. One value,
explicit resolution, and a smart ``AUTO`` default that auto-upgrades to the
shipped classifier once its artifact is bundled — no flag-flipping required.

Values:
- ``AUTO`` (default): local_classifier if a healthy artifact is present, else
  the legacy flag-based selection (edge model / cloud LLM), else heuristic-only.
- ``LOCAL_CLASSIFIER``: force the shipped numpy classifier (FEP-0012).
- ``EDGE``: force the Ollama edge model (tiered).
- ``LLM``: force the cloud LLM decision service.
- ``HEURISTIC``: no model — keyword heuristics only.

This keeps the user-facing config surface to ONE value with sensible defaults,
aligned with first-principles / co-design (one knob, explicit semantics).
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class DecisionBackend(str, Enum):
    """Which backend serves micro-decisions."""

    AUTO = "auto"
    LOCAL_CLASSIFIER = "local_classifier"
    EDGE = "edge"
    LLM = "llm"
    HEURISTIC = "heuristic"

    @classmethod
    def parse(cls, value: Any) -> "DecisionBackend":
        """Coerce a raw config value to a backend (falls back to AUTO)."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.AUTO
