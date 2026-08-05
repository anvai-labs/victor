# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from victor.agent.orchestrator import AgentOrchestrator


def test_orchestrator_removed_chat_compat_diagnostics_surface():
    orchestrator = object.__new__(AgentOrchestrator)

    assert hasattr(orchestrator, "get_deprecated_chat_compat_report") is False
    assert hasattr(orchestrator, "has_deprecated_chat_compat_usage") is False
