# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Architecture guardrails for the canonical chat runtime path."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def _method_source(relative_path: str, class_name: str, method_name: str) -> str:
    source = _read(relative_path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method_name
                ):
                    segment = ast.get_source_segment(source, child)
                    if segment is None:
                        raise AssertionError(
                            f"Could not recover source for {class_name}.{method_name} in {relative_path}"
                        )
                    return segment
    raise AssertionError(f"Method {class_name}.{method_name} not found in {relative_path}")


def _function_source(relative_path: str, function_name: str) -> str:
    source = _read(relative_path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise AssertionError(
                    f"Could not recover source for function {function_name} in {relative_path}"
                )
            return segment
    raise AssertionError(f"Function {function_name} not found in {relative_path}")


def test_orchestrator_public_chat_entrypoints_delegate_to_chat_service() -> None:
    source = _read("victor/agent/orchestrator.py")
    assert "async def _chat_via_agentic_loop" not in source

    chat_source = _method_source("victor/agent/orchestrator.py", "AgentOrchestrator", "chat")
    assert "_chat_service.chat(" in chat_source
    assert "AgenticLoop" not in chat_source

    stream_source = _method_source(
        "victor/agent/orchestrator.py", "AgentOrchestrator", "stream_chat"
    )
    assert "_chat_service.stream_chat(" in stream_source
    assert "AgenticLoop" not in stream_source
    assert "_get_service_streaming_runtime" not in stream_source


def test_chat_service_no_longer_owns_loop_execution() -> None:
    chat_source = _method_source("victor/agent/services/chat_service.py", "ChatService", "chat")
    stream_source = _method_source(
        "victor/agent/services/chat_service.py", "ChatService", "stream_chat"
    )
    loop_guard_source = _method_source(
        "victor/agent/services/chat_service.py", "ChatService", "_run_agentic_loop"
    )

    assert "_run_agentic_loop(" not in chat_source
    assert "_run_agentic_loop(" not in stream_source
    assert "raise RuntimeError" in loop_guard_source
    assert "while " not in loop_guard_source
    assert "_get_completion(" not in loop_guard_source


def test_deprecated_chat_shim_modules_are_removed() -> None:
    source = _read("victor/agent/runtime/bootstrapper.py")

    assert "chat_compat.py" not in source
    assert "sync_chat_compat.py" not in source
    assert "streaming_chat_compat.py" not in source
    assert "unified_chat_compat.py" not in source


def test_deprecated_sync_chat_coordinator_is_not_wired_to_turn_executor() -> None:
    source = _read("victor/agent/orchestrator_properties.py")

    assert "_ensure_sync_chat_coordinator" not in source
    assert "turn_executor=self.turn_executor" not in source


def test_deprecated_unified_chat_coordinator_is_not_wired_to_nested_chat_shims() -> None:
    source = _read("victor/agent/orchestrator_properties.py")

    assert "_ensure_unified_chat_coordinator" not in source
    assert "_ensure_sync_chat_coordinator(self)" not in source
    assert "_ensure_streaming_chat_coordinator(self)" not in source


def test_runtime_bootstrapper_does_not_bind_deprecated_chat_shim_getters_to_facade() -> None:
    source = _method_source(
        "victor/agent/runtime/bootstrapper.py",
        "AgentRuntimeBootstrapper",
        "create_facades",
    )
    assert "deprecated_chat_coordinator=" not in source
    assert "get_chat_coordinator=" not in source
    assert "get_tool_coordinator=" not in source
    assert "get_session_coordinator=" not in source
    assert "get_sync_chat_coordinator=" not in source
    assert "get_streaming_chat_coordinator=" not in source
    assert "get_unified_chat_coordinator=" not in source


def test_orchestrator_tool_capability_check_delegates_to_provider_runtime() -> None:
    source = _method_source(
        "victor/agent/orchestrator.py",
        "AgentOrchestrator",
        "_model_supports_tool_calls",
    )

    assert "_get_provider_management_runtime().model_supports_tool_calls()" in source
    assert "ollama_capability_detector" not in source
    assert "is_tool_call_supported" not in source


def test_orchestrator_tool_output_formatting_delegates_to_tool_runtime() -> None:
    source = _method_source(
        "victor/agent/orchestrator.py",
        "AgentOrchestrator",
        "_format_tool_output",
    )

    assert "_get_tool_execution_runtime().format_tool_output(" in source
    assert "FormattingContext" not in source
    assert "_conversation_controller.get_context_metrics" not in source


def test_orchestrator_cache_flag_computation_delegates_to_prompt_runtime() -> None:
    source = _method_source(
        "victor/agent/orchestrator.py",
        "AgentOrchestrator",
        "_compute_cache_flags",
    )

    assert "_get_prompt_builder_runtime().compute_cache_flags()" in source
    assert "supports_prompt_caching" not in source
    assert "supports_kv_prefix_caching" not in source
