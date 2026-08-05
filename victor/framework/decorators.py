# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple decorator API for defining agents and tasks.

Reduces boilerplate for the 90% case: convert a function into an agent
or task with a single decorator line.

Usage::

    import victor

    @victor.agent(provider="anthropic", tools=["filesystem", "git"])
    async def code_reviewer(prompt: str) -> str:
        \"\"\"You are a senior code reviewer focused on security and correctness.\"\"\"
        pass  # Framework handles implementation

    # Calling the decorated function runs the agent
    result = await code_reviewer("Review auth.py for SQL injection risks")
    print(result.content)

    # Tasks carry metadata for composition
    @victor.task(description="Analyse code quality", expected_output="Markdown report")
    async def quality_check(agent, file_path: str):
        return await agent.run(f"Analyse {file_path} for quality issues")
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.agent import Agent, TaskResult
    from victor.framework.tools import ToolsInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentCallable — the object returned by @victor.agent
# ---------------------------------------------------------------------------


class AgentCallable:
    """Callable wrapper that lazily creates an Agent and forwards calls to it.

    The underlying Agent is created on the first call and reused for subsequent
    calls (session-scoped singleton). Call ``reset()`` to force re-creation.

    Attributes:
        __name__: Function name (for introspection / repr)
        __doc__: System prompt derived from the decorated function's docstring
    """

    def __init__(
        self,
        func: Callable,
        provider: str,
        model: Optional[str],
        tools: "ToolsInput",
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        thinking: bool,
        airgapped: bool,
        profile: Optional[str],
        workspace: Optional[str],
        extra_kwargs: Dict[str, Any],
    ) -> None:
        self._func = func
        self._provider = provider
        self._model = model
        self._tools = tools
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or (
            inspect.cleandoc(func.__doc__) if func.__doc__ else None
        )
        self._thinking = thinking
        self._airgapped = airgapped
        self._profile = profile
        self._workspace = workspace
        self._extra_kwargs = extra_kwargs

        self._agent: Optional["Agent"] = None

        # Preserve function identity
        functools.update_wrapper(self, func)

    async def _get_agent(self) -> "Agent":
        """Lazy-create the Agent on first use."""
        if self._agent is None:
            from victor.framework.agent import Agent

            self._agent = await Agent.create(
                provider=self._provider,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=self._tools,
                thinking=self._thinking,
                airgapped=self._airgapped,
                profile=self._profile,
                workspace=self._workspace,
                **self._extra_kwargs,
            )

            logger.debug(
                "AgentCallable '%s': agent created (provider=%s, model=%s)",
                self.__name__,
                self._provider,
                self._model,
            )
        return self._agent

    async def __call__(
        self, prompt: str, *, context: Optional[Dict[str, Any]] = None
    ) -> "TaskResult":
        """Run the agent with the given prompt.

        Args:
            prompt: What the agent should do
            context: Optional dict of context variables (files, error messages, etc.)

        Returns:
            TaskResult with content, tool_calls, success flag, and metadata
        """
        agent = await self._get_agent()
        if not self._system_prompt:
            return await agent.run(prompt, context=context)

        from victor.agent.prompt_pipeline import prompt_overlay_runtime_overrides

        runtime_context_overrides = prompt_overlay_runtime_overrides(
            f"decorator.{self.__name__}.system_prompt",
            self._system_prompt,
        )
        return await agent.run(
            prompt,
            context=context,
            runtime_context_overrides=runtime_context_overrides,
        )

    def reset(self) -> None:
        """Force the agent to be re-created on the next call.

        Useful in tests or when you need a fresh conversation history.
        """
        self._agent = None

    @property
    def agent(self) -> Optional["Agent"]:
        """Return the underlying Agent if already created, else None."""
        return self._agent

    def __repr__(self) -> str:
        status = "ready" if self._agent else "lazy"
        return (
            f"<AgentCallable '{self.__name__}' provider={self._provider!r} "
            f"model={self._model!r} status={status}>"
        )


# ---------------------------------------------------------------------------
# TaskDefinition — the object returned by @victor.task
# ---------------------------------------------------------------------------


class TaskDefinition:
    """Callable wrapper that carries task metadata for agent composition.

    Wraps an async function with description, expected output, and optional
    tool hints. The wrapped function is called as-is; the metadata is
    accessible for team/pipeline coordinators.

    Example::

        @victor.task(description="Generate unit tests", expected_output="pytest file")
        async def generate_tests(agent, source_file: str) -> TaskResult:
            return await agent.run(f"Write pytest tests for {source_file}")

        # In a team scenario the coordinator can inspect task.description
        print(generate_tests.description)
    """

    def __init__(
        self,
        func: Callable,
        description: str,
        expected_output: str,
        tools: Optional[List[str]],
        context_vars: Optional[List[str]],
        async_execution: bool,
    ) -> None:
        self._func = func
        self.description = description
        self.expected_output = expected_output
        self.tools = tools or []
        self.context_vars = context_vars or []
        self.async_execution = async_execution

        functools.update_wrapper(self, func)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the task function."""
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(*args, **kwargs)
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<TaskDefinition '{self.__name__}' " f"description={self.description!r}>"


# ---------------------------------------------------------------------------
# Public decorator functions
# ---------------------------------------------------------------------------


def agent(
    provider: str = "anthropic",
    model: Optional[str] = None,
    *,
    tools: "ToolsInput" = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
    thinking: bool = False,
    airgapped: bool = False,
    profile: Optional[str] = None,
    workspace: Optional[str] = None,
    **kwargs: Any,
) -> Callable[[Callable], AgentCallable]:
    """Decorator that converts a function into a reusable agent.

    The decorated function's docstring becomes the system prompt unless
    ``system_prompt`` is provided explicitly. The function body is ignored —
    only its signature and docstring are used.

    Args:
        provider: LLM provider (anthropic, openai, ollama, …). Default: "anthropic".
        model: Model identifier. None uses the provider default.
        tools: Tool configuration — ToolSet, list of category names, or None.
        temperature: Sampling temperature (0.0–1.0). Default: 0.7.
        max_tokens: Maximum tokens to generate. Default: 4096.
        system_prompt: Explicit system prompt. Overrides the function docstring.
        thinking: Enable extended thinking mode (Claude only). Default: False.
        airgapped: Disable network-dependent tools. Default: False.
        profile: Profile name from ~/.victor/profiles.yaml.
        workspace: Working directory for file operations.
        **kwargs: Extra arguments forwarded to Agent.create().

    Returns:
        AgentCallable — an async-callable that runs the agent and returns TaskResult.

    Example::

        @victor.agent(provider="anthropic", tools=["filesystem", "git"])
        async def dev_agent(prompt: str) -> str:
            \"\"\"You are a senior software engineer. Be concise and precise.\"\"\"
            pass  # Framework handles implementation

        result = await dev_agent("Add type hints to utils.py")
        print(result.content)
    """

    def decorator(func: Callable) -> AgentCallable:
        return AgentCallable(
            func=func,
            provider=provider,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            thinking=thinking,
            airgapped=airgapped,
            profile=profile,
            workspace=workspace,
            extra_kwargs=kwargs,
        )

    return decorator


def task(
    description: str,
    expected_output: str = "",
    *,
    tools: Optional[List[str]] = None,
    context_vars: Optional[List[str]] = None,
    async_execution: bool = False,
) -> Callable[[Callable], TaskDefinition]:
    """Decorator that adds task metadata to an async function.

    Used for composition in multi-agent pipelines. The function still
    runs normally when called; the decorator enriches it with metadata
    that coordinators and team formations can inspect.

    Args:
        description: Human-readable description of what the task does.
        expected_output: Description of the expected result format.
        tools: Optional list of tool names this task requires.
        context_vars: Names of context variables the task reads.
        async_execution: Hint to the coordinator that this task can run
            concurrently with others. Default: False.

    Returns:
        TaskDefinition — callable with .description, .expected_output, .tools attrs.

    Example::

        @victor.task(
            description="Write unit tests for a Python module",
            expected_output="A pytest file with at least 80% coverage",
            tools=["filesystem"],
        )
        async def write_tests(agent, module_path: str) -> TaskResult:
            return await agent.run(f"Write pytest tests for {module_path}")

        # Metadata is available for introspection
        print(write_tests.description)
        print(write_tests.expected_output)
    """

    def decorator(func: Callable) -> TaskDefinition:
        return TaskDefinition(
            func=func,
            description=description,
            expected_output=expected_output,
            tools=tools,
            context_vars=context_vars,
            async_execution=async_execution,
        )

    return decorator
