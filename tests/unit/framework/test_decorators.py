# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for @victor.agent and @victor.task decorator API."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.decorators import (
    AgentCallable,
    TaskDefinition,
    agent,
    task,
)

# ---------------------------------------------------------------------------
# @victor.agent tests
# ---------------------------------------------------------------------------


class TestAgentDecorator:
    """@agent decorator creates AgentCallable instances."""

    def test_returns_agent_callable(self):
        """@agent wraps a function in AgentCallable."""

        @agent(provider="anthropic")
        async def my_agent(prompt: str) -> str:
            """You are a helpful assistant."""
            pass  # Framework handles implementation

        assert isinstance(my_agent, AgentCallable)

    def test_preserves_function_name(self):
        """AgentCallable.__name__ matches the decorated function."""

        @agent(provider="anthropic")
        async def code_reviewer(prompt: str) -> str:
            """Review code."""
            pass  # Framework handles implementation

        assert code_reviewer.__name__ == "code_reviewer"

    def test_docstring_becomes_system_prompt(self):
        """Decorated function docstring is used as system prompt."""

        @agent(provider="anthropic")
        async def helper(prompt: str) -> str:
            """You are an expert code reviewer."""
            pass  # Framework handles implementation

        assert helper._system_prompt == "You are an expert code reviewer."

    def test_explicit_system_prompt_overrides_docstring(self):
        """Explicit system_prompt kwarg overrides docstring."""

        @agent(provider="anthropic", system_prompt="Custom prompt")
        async def helper(prompt: str) -> str:
            """Ignored docstring."""
            pass  # Framework handles implementation

        assert helper._system_prompt == "Custom prompt"

    def test_no_docstring_system_prompt_is_none(self):
        """No docstring → _system_prompt is None."""

        @agent(provider="openai")
        async def bare(prompt: str) -> str:
            pass  # Framework handles implementation

        assert bare._system_prompt is None

    def test_provider_stored(self):
        """Provider argument is stored on AgentCallable."""

        @agent(provider="ollama", model="llama3")
        async def local_agent(prompt: str) -> str:
            pass  # Framework handles implementation

        assert local_agent._provider == "ollama"
        assert local_agent._model == "llama3"

    def test_tools_stored(self):
        """tools argument is stored."""

        @agent(provider="anthropic", tools=["filesystem", "git"])
        async def fs_agent(prompt: str) -> str:
            pass  # Framework handles implementation

        assert fs_agent._tools == ["filesystem", "git"]

    def test_agent_is_none_before_first_call(self):
        """Underlying agent is lazy — None until first __call__."""

        @agent(provider="anthropic")
        async def lazy(prompt: str) -> str:
            pass  # Framework handles implementation

        assert lazy.agent is None

    def test_reset_clears_agent(self):
        """reset() sets _agent back to None."""

        @agent(provider="anthropic")
        async def resettable(prompt: str) -> str:
            pass  # Framework handles implementation

        resettable._agent = MagicMock()  # Simulate a created agent
        resettable.reset()
        assert resettable.agent is None

    def test_repr_contains_provider_and_name(self):
        """repr shows provider and function name."""

        @agent(provider="anthropic")
        async def my_agent(prompt: str) -> str:
            pass  # Framework handles implementation

        r = repr(my_agent)
        assert "my_agent" in r
        assert "anthropic" in r

    async def test_call_creates_agent_and_runs(self):
        """Calling the AgentCallable creates Agent and calls run()."""
        mock_result = MagicMock()
        mock_result.content = "test result"

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.get_orchestrator.return_value = MagicMock()

        @agent(provider="anthropic")
        async def my_agent(prompt: str) -> str:
            pass  # Framework handles implementation

        with patch(
            "victor.framework.agent.Agent.create",
            new=AsyncMock(return_value=mock_agent),
        ):
            result = await my_agent("do something")

        mock_agent.run.assert_called_once_with("do something", context=None)
        assert result.content == "test result"

    async def test_agent_is_reused_on_second_call(self):
        """The underlying agent is created once and reused."""
        mock_result = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.get_orchestrator.return_value = MagicMock()

        @agent(provider="anthropic")
        async def cached(prompt: str) -> str:
            pass  # Framework handles implementation

        with patch(
            "victor.framework.agent.Agent.create",
            new=AsyncMock(return_value=mock_agent),
        ) as mock_create:
            await cached("first")
            await cached("second")

        # Agent.create called only once
        mock_create.assert_called_once()
        assert mock_agent.run.call_count == 2

    async def test_context_forwarded_to_run(self):
        """context kwarg is forwarded to agent.run()."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock())
        mock_agent.get_orchestrator.return_value = MagicMock()

        @agent(provider="anthropic")
        async def ctx_agent(prompt: str) -> str:
            pass  # Framework handles implementation

        ctx = {"file": "auth.py"}
        with patch(
            "victor.framework.agent.Agent.create",
            new=AsyncMock(return_value=mock_agent),
        ):
            await ctx_agent("fix it", context=ctx)

        mock_agent.run.assert_called_once_with("fix it", context=ctx)

    async def test_system_prompt_passed_as_scoped_prompt_overlay(self):
        """Docstring system prompts are scoped to the decorated call."""
        mock_orch = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock())
        mock_agent.get_orchestrator.return_value = mock_orch

        @agent(provider="anthropic")
        async def doc_agent(prompt: str) -> str:
            """You are a documentation writer."""
            pass  # Framework handles implementation

        with patch(
            "victor.framework.agent.Agent.create",
            new=AsyncMock(return_value=mock_agent),
        ):
            await doc_agent("write docs")

        mock_orch.set_system_prompt.assert_not_called()
        mock_agent.run.assert_called_once_with(
            "write docs",
            context=None,
            runtime_context_overrides={
                "prompt_overlays": [
                    {
                        "name": "decorator.doc_agent.system_prompt",
                        "content": "You are a documentation writer.",
                        "placement": "turn_prefix",
                    }
                ]
            },
        )

    def test_importable_from_victor_namespace(self):
        """@victor.agent must be accessible from the top-level victor namespace."""
        import victor

        assert hasattr(victor, "agent"), "victor.agent decorator not exported"
        assert callable(victor.agent)


# ---------------------------------------------------------------------------
# @victor.task tests
# ---------------------------------------------------------------------------


class TestTaskDecorator:
    """@task decorator creates TaskDefinition instances."""

    def test_returns_task_definition(self):
        """@task wraps a function in TaskDefinition."""

        @task(description="Generate tests", expected_output="pytest file")
        async def gen_tests(agent, path: str):
            pass  # Framework handles implementation

        assert isinstance(gen_tests, TaskDefinition)

    def test_description_stored(self):
        """description is stored as attribute."""

        @task(description="Write docs", expected_output="Markdown file")
        async def write_docs(agent, module: str):
            pass  # Framework handles implementation

        assert write_docs.description == "Write docs"

    def test_expected_output_stored(self):
        """expected_output is stored as attribute."""

        @task(description="Analyse", expected_output="JSON report")
        async def analyse(agent, src: str):
            pass  # Framework handles implementation

        assert analyse.expected_output == "JSON report"

    def test_tools_default_empty(self):
        """tools defaults to empty list when not provided."""

        @task(description="Simple task")
        async def simple(agent):
            pass  # Framework handles implementation

        assert simple.tools == []

    def test_tools_stored(self):
        """tools list is stored on TaskDefinition."""

        @task(description="FS task", tools=["filesystem", "git"])
        async def fs_task(agent):
            pass  # Framework handles implementation

        assert fs_task.tools == ["filesystem", "git"]

    def test_preserves_function_name(self):
        """__name__ matches the decorated function."""

        @task(description="Named task")
        async def named_fn(agent):
            pass  # Framework handles implementation

        assert named_fn.__name__ == "named_fn"

    async def test_callable_invokes_function(self):
        """Calling TaskDefinition invokes the underlying async function."""
        calls = []

        @task(description="Record call")
        async def recorder(agent, value: int):
            calls.append(value)
            return value * 2

        result = await recorder(None, 21)
        assert result == 42
        assert calls == [21]

    def test_async_execution_flag(self):
        """async_execution flag is stored."""

        @task(description="Concurrent task", async_execution=True)
        async def concurrent(agent):
            pass  # Framework handles implementation

        assert concurrent.async_execution is True

    def test_importable_from_victor_namespace(self):
        """@victor.task must be accessible from the top-level victor namespace."""
        import victor

        assert hasattr(victor, "task"), "victor.task decorator not exported"
        assert callable(victor.task)

    def test_repr_contains_description_and_name(self):
        """repr shows function name and description."""

        @task(description="My important task")
        async def my_task(agent):
            pass  # Framework handles implementation

        r = repr(my_task)
        assert "my_task" in r
        assert "My important task" in r
