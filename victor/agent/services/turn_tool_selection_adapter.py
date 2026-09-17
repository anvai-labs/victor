# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Adapt split turn contexts to the shared tool-selection runtime (FEP-0034).

Turn inputs belong to this adapter; optional session state belongs to the
orchestrator when present. Keeping the adapter for the executor's lifetime also
preserves KV and exploration state for standalone protocol hosts.
"""

from __future__ import annotations

from typing import Any


class TurnToolSelectionAdapter:
    _CONTEXT_ATTRIBUTES = {
        "settings": "_chat_context",
        "conversation": "_chat_context",
        "messages": "_chat_context",
        "tool_selector": "_tool_context",
        "use_semantic_selection": "_tool_context",
        "observed_files": "_tool_context",
        "provider": "_provider_context",
    }
    _SESSION_STATE = frozenset({"_e3tir_reranker", "_kv_frozen_tools"})

    def __init__(self, executor: Any) -> None:
        self._executor = executor
        self._current_user_message = ""
        self._current_intent = None
        self._fallback_planner = None

    def bind_turn(self, user_message: str, intent: Any) -> None:
        from victor.agent.action_authorizer import ActionIntent

        self._current_user_message = user_message
        if intent is None:
            intent = getattr(self._executor._resolve_orchestrator(), "_current_intent", None)
        try:
            self._current_intent = ActionIntent(intent) if intent is not None else None
        except ValueError:
            self._current_intent = None

    def __getattr__(self, name: str) -> Any:
        context_name = self._CONTEXT_ATTRIBUTES.get(name)
        if context_name:
            return getattr(getattr(self._executor, context_name), name)
        owner = self._executor._resolve_orchestrator()
        if name == "tools":
            return getattr(self.tool_selector, "tools", None)
        return getattr(owner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._SESSION_STATE:
            owner = self._executor._resolve_orchestrator()
            if owner is not None:
                setattr(owner, name, value)
                return
        object.__setattr__(self, name, value)

    @property
    def _tool_planner(self) -> Any:
        planner = getattr(self._executor._tool_context, "_tool_planner", None)
        if planner is not None:
            return planner
        if self._fallback_planner is None:
            from victor.agent.services.tool_planning_runtime import ToolPlanner

            self._fallback_planner = ToolPlanner(
                getattr(self, "tool_registrar", None), self.settings
            )
        return self._fallback_planner

    def _model_supports_tool_calls(self) -> bool:
        check = getattr(self._executor._tool_context, "_model_supports_tool_calls", None)
        return check() if callable(check) else True

    def _tool_skip_mode(self, message: str) -> str:
        owner = self._executor._resolve_orchestrator()
        check = getattr(owner, "_tool_skip_mode", None)
        if callable(check):
            return check(message)
        from victor.agent.tool_supply_policy import classify_tool_supply

        return classify_tool_supply(message, edge_check=lambda _message, _confidence: False)
