# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Classification backend configuration (mirrors FEP-0012 ``DecisionServiceSettings``).

``classification_backend`` selects whether classification triage routes through
the ``TieredDecisionService``. ``AUTO`` (default) defers to the
``USE_TIERED_CLASSIFICATION`` feature flag (prior behavior); ``TIERED`` forces it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from victor.agent.services.classification_backend import ClassificationBackend


class ClassificationSettings(BaseModel):
    """Classification triage configuration.

    Attributes:
        classification_backend: ``auto`` (defer to USE_TIERED_CLASSIFICATION) or
            ``tiered``.
    """

    classification_backend: ClassificationBackend = Field(
        default=ClassificationBackend.AUTO,
        description="Classification triage backend: auto|tiered",
    )
