# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.orchestrator import AgentOrchestrator


def test_emit_prompt_used_event_delegates_only_to_prompt_pipeline() -> None:
    pipeline = MagicMock()
    orchestrator = SimpleNamespace(_prompt_pipeline=pipeline)

    AgentOrchestrator._emit_prompt_used_event(orchestrator, "prompt body")

    pipeline._emit_prompt_used_event.assert_called_once_with("prompt body")


def test_emit_prompt_used_event_is_no_op_without_pipeline() -> None:
    orchestrator = SimpleNamespace()

    AgentOrchestrator._emit_prompt_used_event(orchestrator, "prompt body")
