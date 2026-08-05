# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""AgenticLoop executor configuration (mirrors FEP-0012 ``DecisionServiceSettings``).

``executor`` selects the ``AgenticLoop`` run/stream executor. ``AUTO`` (default)
defers to the ``USE_STATEGRAPH_AGENTIC_LOOP`` feature flag (prior behavior);
``STATEGRAPH`` forces the StateGraph executor.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from victor.framework.agentic_loop_executor import AgenticLoopExecutor


class AgenticLoopSettings(BaseModel):
    """AgenticLoop execution configuration.

    Attributes:
        executor: ``auto`` (defer to USE_STATEGRAPH_AGENTIC_LOOP) or ``stategraph``.
    """

    executor: AgenticLoopExecutor = Field(
        default=AgenticLoopExecutor.AUTO,
        description="AgenticLoop executor: auto|stategraph",
    )
