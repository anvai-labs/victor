# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""AgenticLoop executor configuration (mirrors FEP-0012 ``DecisionServiceSettings``).

``executor`` selects the ``AgenticLoop`` run/stream executor. ``AUTO`` (default)
defers to the ``USE_STATEGRAPH_AGENTIC_LOOP`` feature flag (prior behavior);
``STANDARD`` pins the standard loop; ``STATEGRAPH`` forces StateGraph.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enum import Enum


class AgenticLoopExecutor(str, Enum):
    """Configured executor; AUTO resolves to STANDARD or STATEGRAPH before admission."""

    AUTO = "auto"
    STANDARD = "standard"
    STATEGRAPH = "stategraph"

    @classmethod
    def parse(cls, value: object) -> "AgenticLoopExecutor":
        """Validate an explicit choice; malformed configuration never changes executor."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError("Agentic loop executor must be auto, standard or stategraph")


class AgenticLoopSettings(BaseModel):
    """AgenticLoop execution configuration.

    Attributes:
        executor: ``auto`` (legacy flag), ``standard`` or ``stategraph``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    executor: AgenticLoopExecutor = Field(
        default=AgenticLoopExecutor.AUTO,
        description="AgenticLoop executor: auto|standard|stategraph",
    )

    @field_validator("executor", mode="before")
    @classmethod
    def validate_executor(cls, value: object) -> AgenticLoopExecutor:
        return AgenticLoopExecutor.parse(value)
