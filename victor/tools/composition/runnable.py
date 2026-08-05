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

"""LCEL-style tool composition for functional tool chaining.

This module provides LangChain Expression Language (LCEL) style composition
patterns for Victor's tool system, enabling:

- Pipe-based chaining: `tool1 | tool2 | tool3`
- Parallel execution: `RunnableParallel(a=tool1, b=tool2)`
- Conditional routing: `RunnableBranch((cond1, tool1), (cond2, tool2), default=tool3)`
- Result transformation: `tool | transform_fn`

Design Principles:
- Tools remain standalone and backward compatible
- Composition creates new Runnable objects without modifying originals
- Async-first with sync fallback support
- Results flow through chains via output mapping
- Compatible with existing ToolPipeline infrastructure

Example usage:
    from victor.tools.composition import (
        RunnableParallel,
        RunnableBranch,
        RunnablePassthrough,
        as_runnable,
    )
    from victor.tools.filesystem import read, ls

    # Simple pipe chain
    chain = as_runnable(ls) | as_runnable(read)

    # Parallel execution
    parallel = RunnableParallel(
        files=as_runnable(ls),
        config=as_runnable(read, path="config.yaml"),
    )

    # Conditional routing
    branch = RunnableBranch(
        (lambda x: x.get("type") == "python", as_runnable(python_lint)),
        (lambda x: x.get("type") == "javascript", as_runnable(js_lint)),
        default=as_runnable(generic_check),
    )

    # Execute
    result = await chain.invoke({"path": "."})
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from victor.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# Type variables for generic composition
Input = TypeVar("Input")
Output = TypeVar("Output")
Other = TypeVar("Other")


@dataclass
class RunnableConfig:
    """Configuration for runnable execution.

    Attributes:
        tags: Tags for tracing/logging
        metadata: Additional metadata passed through execution
        max_concurrency: Maximum parallel tasks for RunnableParallel
        timeout: Timeout in seconds for execution
        callbacks: Optional callbacks for execution events
    """

    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_concurrency: int = 10
    timeout: Optional[float] = None
    callbacks: Optional[List[Callable]] = None


class Runnable(ABC, Generic[Input, Output]):
    """Base class for composable runnables in LCEL-style chains.

    This is the core abstraction for composable tool execution. All runnables
    implement `invoke` for single execution and support the `|` operator for
    chaining.

    Type Parameters:
        Input: The input type for invoke()
        Output: The output type from invoke()

    Example:
        class MyRunnable(Runnable[Dict, str]):
            async def invoke(self, input: Dict, config: RunnableConfig = None) -> str:
                return f"Processed: {input}"

        # Use in chain
        chain = runnable1 | runnable2 | MyRunnable()
    """

    @abstractmethod
    async def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Output:
        """Execute this runnable with the given input.

        Args:
            input: Input data for execution
            config: Optional configuration for execution

        Returns:
            Output from execution
        """
        ...

    def __or__(
        self, other: Union["Runnable[Output, Other]", Callable[[Output], Other]]
    ) -> "RunnableSequence[Input, Other]":
        """Chain this runnable with another using the pipe operator.

        Args:
            other: Next runnable or callable in the chain

        Returns:
            A RunnableSequence combining both runnables

        Example:
            chain = read_tool | analyze_tool | format_tool
        """
        if isinstance(other, Runnable):
            return RunnableSequence([self, other])
        elif callable(other):
            return RunnableSequence([self, RunnableLambda(other)])
        raise TypeError(f"Cannot chain Runnable with {type(other)}")

    def __ror__(
        self, other: Union["Runnable[Other, Input]", Callable[[Other], Input]]
    ) -> "RunnableSequence[Other, Output]":
        """Support reverse pipe operator for left-side non-runnables.

        Args:
            other: Previous runnable or callable

        Returns:
            A RunnableSequence with other first
        """
        if isinstance(other, Runnable):
            return RunnableSequence([other, self])
        elif callable(other):
            return RunnableSequence([RunnableLambda(other), self])
        raise TypeError(f"Cannot chain {type(other)} with Runnable")

    def pipe(self, *others: Union["Runnable", Callable]) -> "RunnableSequence":
        """Explicitly pipe through multiple runnables.

        Alternative to chained `|` operators for programmatic composition.

        Args:
            *others: Sequence of runnables or callables to chain

        Returns:
            A RunnableSequence of all runnables

        Example:
            chain = read_tool.pipe(analyze, format, output)
        """
        runnables: List[Runnable] = [self]
        for other in others:
            if isinstance(other, Runnable):
                runnables.append(other)
            elif callable(other):
                runnables.append(RunnableLambda(other))
            else:
                raise TypeError(f"Cannot pipe with {type(other)}")
        return RunnableSequence(runnables)

    def bind(self, **kwargs: Any) -> "RunnableBinding[Input, Output]":
        """Bind arguments to this runnable.

        Creates a new runnable with pre-set arguments that will be merged
        with input at execution time.

        Args:
            **kwargs: Arguments to bind

        Returns:
            A RunnableBinding with bound arguments

        Example:
            read_config = read_tool.bind(path="config.yaml")
            result = await read_config.invoke({})
        """
        return RunnableBinding(self, kwargs)

    def with_config(self, config: RunnableConfig) -> "RunnableBinding[Input, Output]":
        """Create a runnable with fixed configuration.

        Args:
            config: Configuration to use for all invocations

        Returns:
            A RunnableBinding with fixed config
        """
        return RunnableBinding(self, {}, config)

    async def batch(
        self,
        inputs: Sequence[Input],
        config: Optional[RunnableConfig] = None,
    ) -> List[Output]:
        """Execute this runnable on multiple inputs.

        Default implementation runs sequentially. Subclasses may override
        for optimized batch execution.

        Args:
            inputs: Sequence of inputs to process
            config: Optional shared configuration

        Returns:
            List of outputs in same order as inputs
        """
        results = []
        for inp in inputs:
            result = await self.invoke(inp, config)
            results.append(result)
        return results


class RunnableSequence(Runnable[Input, Output]):
    """A sequence of runnables executed in order.

    Each runnable's output becomes the next runnable's input.
    Created automatically by the `|` operator.

    Example:
        # These are equivalent:
        chain = tool1 | tool2 | tool3
        chain = RunnableSequence([tool1, tool2, tool3])
    """

    def __init__(self, runnables: List[Runnable]):
        """Initialize with a list of runnables.

        Args:
            runnables: Ordered list of runnables to execute
        """
        if not runnables:
            raise ValueError("RunnableSequence requires at least one runnable")
        self._runnables = runnables

    @property
    def first(self) -> Runnable:
        """Get the first runnable in the sequence."""
        return self._runnables[0]

    @property
    def last(self) -> Runnable:
        """Get the last runnable in the sequence."""
        return self._runnables[-1]

    @property
    def middle(self) -> List[Runnable]:
        """Get middle runnables (excluding first and last)."""
        return self._runnables[1:-1] if len(self._runnables) > 2 else []

    async def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Output:
        """Execute all runnables in sequence.

        Args:
            input: Input to first runnable
            config: Configuration passed to all runnables

        Returns:
            Output from last runnable
        """
        current = input
        for runnable in self._runnables:
            current = await runnable.invoke(current, config)
        return current

    def __or__(
        self, other: Union[Runnable[Output, Other], Callable[[Output], Other]]
    ) -> "RunnableSequence[Input, Other]":
        """Extend the sequence with another runnable."""
        if isinstance(other, RunnableSequence):
            # Flatten nested sequences
            return RunnableSequence(self._runnables + other._runnables)
        elif isinstance(other, Runnable):
            return RunnableSequence(self._runnables + [other])
        elif callable(other):
            return RunnableSequence(self._runnables + [RunnableLambda(other)])
        raise TypeError(f"Cannot chain with {type(other)}")

    def __repr__(self) -> str:
        names = [getattr(r, "name", type(r).__name__) for r in self._runnables]
        return f"RunnableSequence({' | '.join(names)})"


class RunnableParallel(Runnable[Input, Dict[str, Any]]):
    """Execute multiple runnables in parallel and collect results.

    All runnables receive the same input and execute concurrently.
    Results are collected into a dictionary keyed by the names provided.

    Example:
        parallel = RunnableParallel(
            summary=analyze_tool,
            security=security_scan_tool,
            style=style_check_tool,
        )
        result = await parallel.invoke({"path": "main.py"})
        # result = {"summary": ..., "security": ..., "style": ...}
    """

    def __init__(
        self,
        steps: Optional[Mapping[str, Runnable]] = None,
        **kwargs: Runnable,
    ):
        """Initialize with named runnables.

        Args:
            steps: Mapping of names to runnables
            **kwargs: Alternative way to specify named runnables
        """
        self._steps: Dict[str, Runnable] = {}
        if steps:
            self._steps.update(steps)
        self._steps.update(kwargs)

        if not self._steps:
            raise ValueError("RunnableParallel requires at least one step")

    async def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Execute all runnables in parallel.

        Args:
            input: Input passed to all runnables
            config: Configuration for all runnables

        Returns:
            Dictionary mapping step names to their outputs
        """
        config = config or RunnableConfig()

        async def run_step(name: str, runnable: Runnable) -> Tuple[str, Any]:
            try:
                result = await runnable.invoke(input, config)
                return (name, result)
            except Exception as e:
                logger.error(f"Parallel step '{name}' failed: {e}")
                return (name, {"error": str(e), "success": False})

        # Create tasks with optional concurrency limit
        semaphore = asyncio.Semaphore(config.max_concurrency)

        async def bounded_run(name: str, runnable: Runnable) -> Tuple[str, Any]:
            async with semaphore:
                return await run_step(name, runnable)

        tasks = [bounded_run(name, runnable) for name, runnable in self._steps.items()]

        if config.timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=config.timeout,
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build output dictionary
        output = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel execution error: {result}")
                continue
            name, value = result
            output[name] = value

        return output

    def __repr__(self) -> str:
        return f"RunnableParallel({list(self._steps.keys())})"


class RunnableBranch(Runnable[Input, Output]):
    """Conditional routing to different runnables based on input.

    Evaluates conditions in order and routes to the first matching branch.
    Falls back to default if no conditions match.

    Example:
        branch = RunnableBranch(
            (lambda x: x.get("lang") == "python", python_linter),
            (lambda x: x.get("lang") == "javascript", js_linter),
            default=generic_linter,
        )
        result = await branch.invoke({"lang": "python", "code": "..."})
    """

    def __init__(
        self,
        *branches: Tuple[Callable[[Input], bool], Runnable[Input, Output]],
        default: Optional[Runnable[Input, Output]] = None,
    ):
        """Initialize with condition-runnable pairs.

        Args:
            *branches: Tuples of (condition, runnable) evaluated in order
            default: Runnable to use if no conditions match
        """
        self._branches: List[Tuple[Callable[[Input], bool], Runnable[Input, Output]]] = list(
            branches
        )
        self._default = default

        if not self._branches and not self._default:
            raise ValueError("RunnableBranch requires at least one branch or default")

    async def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Output:
        """Route to appropriate runnable based on conditions.

        Args:
            input: Input to evaluate and pass to selected runnable
            config: Configuration for execution

        Returns:
            Output from selected runnable

        Raises:
            ValueError: If no branch matches and no default provided
        """
        for condition, runnable in self._branches:
            try:
                # Support both sync and async conditions
                if asyncio.iscoroutinefunction(condition):
                    matches = await condition(input)
                else:
                    matches = condition(input)

                if matches:
                    return await runnable.invoke(input, config)
            except Exception as e:
                logger.warning(f"Branch condition evaluation failed: {e}")
                continue

        if self._default is not None:
            return await self._default.invoke(input, config)

        raise ValueError("No branch matched and no default provided")

    def __repr__(self) -> str:
        return (
            f"RunnableBranch({len(self._branches)} branches, default={self._default is not None})"
        )


class RunnableLambda(Runnable[Input, Output]):
    """Wrap a callable function as a Runnable.

    Useful for including transformation functions in chains.
    Supports both sync and async functions.

    Example:
        transform = RunnableLambda(lambda x: x["output"].upper())
        chain = read_tool | transform | output_tool
    """

    def __init__(
        self,
        func: Union[Callable[[Input], Output], Callable[[Input], Awaitable[Output]]],
        name: Optional[str] = None,
    ):
        """Initialize with a callable.

        Args:
            func: Sync or async function to wrap
            name: Optional name for debugging
        """
        self._func = func
        self._name = name or getattr(func, "__name__", "lambda")
        self._is_async = asyncio.iscoroutinefunction(func)

    @property
    def name(self) -> str:
        """Get the lambda name."""
        return self._name

    async def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Output:
        """Execute the wrapped function.

        Args:
            input: Input to pass to function
            config: Ignored for lambdas

        Returns:
            Output from function
        """
        if self._is_async:
            return await self._func(input)
        return self._func(input)

    def __repr__(self) -> str:
        return f"RunnableLambda({self._name})"


class RunnablePassthrough(Runnable[Input, Input]):
    """A runnable that passes input through unchanged.

    Useful for creating parallel branches where you want to preserve
    the original input alongside transformed versions.

    Example:
        parallel = RunnableParallel(
            original=RunnablePassthrough(),
            transformed=transform_tool,
        )
    """

    async def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Input:
        """Return input unchanged."""
        return input

    def __repr__(self) -> str:
        return "RunnablePassthrough()"


class RunnableBinding(Runnable[Input, Output]):
    """A runnable with pre-bound arguments.

    Created by calling `.bind()` on a runnable.

    Example:
        bound = read_tool.bind(encoding="utf-8")
        result = await bound.invoke({"path": "file.txt"})
        # Equivalent to: read_tool.invoke({"path": "file.txt", "encoding": "utf-8"})
    """

    def __init__(
        self,
        bound: Runnable[Input, Output],
        kwargs: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ):
        """Initialize with bound runnable and arguments.

        Args:
            bound: The underlying runnable
            kwargs: Arguments to merge with input
            config: Optional fixed configuration
        """
        self._bound = bound
        self._kwargs = kwargs
        self._config = config

    async def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Output:
        """Execute bound runnable with merged arguments.

        Args:
            input: Input to merge with bound kwargs
            config: Configuration (uses bound config if not provided)

        Returns:
            Output from bound runnable
        """
        # Merge input with bound kwargs (input takes precedence)
        if isinstance(input, dict):
            merged = {**self._kwargs, **input}
        else:
            merged = input

        use_config = config or self._config
        return await self._bound.invoke(merged, use_config)

    def __repr__(self) -> str:
        return f"RunnableBinding({self._bound}, kwargs={list(self._kwargs.keys())})"


class ToolRunnable(Runnable[Dict[str, Any], Dict[str, Any]]):
    """Wraps a Victor BaseTool as a Runnable for composition.

    This adapter enables Victor tools to participate in LCEL-style chains
    while maintaining full backward compatibility with the existing tool system.

    The tool's execute() method is called with input as kwargs, and the
    ToolResult is converted to a dictionary for chain compatibility.

    Example:
        from victor.tools.filesystem import read
        from victor.tools.composition import ToolRunnable

        read_runnable = ToolRunnable(read.Tool(read))
        result = await read_runnable.invoke({"path": "file.py"})
    """

    def __init__(
        self,
        tool: "BaseTool",
        output_key: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize with a BaseTool instance.

        Args:
            tool: Victor tool to wrap
            output_key: Key to extract from output (default: use full output)
            input_mapping: Map input keys to tool parameter names
        """
        self._tool = tool
        self._output_key = output_key
        self._input_mapping = input_mapping or {}

    @property
    def name(self) -> str:
        """Get the tool name."""
        return self._tool.name

    @property
    def tool(self) -> "BaseTool":
        """Get the underlying tool."""
        return self._tool

    async def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Execute the tool with input as kwargs.

        Args:
            input: Dictionary of tool arguments
            config: Optional configuration with metadata for context

        Returns:
            Dictionary with tool output and metadata
        """
        # Apply input mapping if specified
        mapped_input = {}
        for key, value in input.items():
            mapped_key = self._input_mapping.get(key, key)
            mapped_input[mapped_key] = value

        # Build execution context from config
        exec_ctx: Dict[str, Any] = {}
        if config:
            exec_ctx["tags"] = config.tags
            exec_ctx["metadata"] = config.metadata

        # Execute tool
        try:
            result: "ToolResult" = await self._tool.execute(exec_ctx, **mapped_input)

            # Convert ToolResult to dict for chain compatibility
            output = {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "metadata": result.metadata or {},
                "tool_name": self._tool.name,
            }

            # Extract specific key if requested
            if self._output_key and isinstance(result.output, dict):
                output["extracted"] = result.output.get(self._output_key)

            return output

        except Exception as e:
            logger.error(f"Tool '{self._tool.name}' execution failed: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "metadata": {"exception": type(e).__name__},
                "tool_name": self._tool.name,
            }

    def __repr__(self) -> str:
        return f"ToolRunnable({self._tool.name})"


class FunctionToolRunnable(Runnable[Dict[str, Any], Dict[str, Any]]):
    """Wraps a @tool decorated function as a Runnable.

    This handles the common case where tools are defined using the @tool
    decorator rather than as BaseTool subclasses.

    Example:
        from victor.tools.filesystem import read

        # read is a decorated function with a .Tool attribute
        read_runnable = FunctionToolRunnable(read)
        result = await read_runnable.invoke({"path": "file.py"})
    """

    def __init__(
        self,
        func: Callable,
        output_key: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize with a @tool decorated function.

        Args:
            func: Function decorated with @tool
            output_key: Key to extract from output
            input_mapping: Map input keys to parameter names
        """
        if not hasattr(func, "Tool"):
            raise TypeError(f"Function {func} is not decorated with @tool")

        self._func = func
        self._tool_instance = func.Tool
        self._output_key = output_key
        self._input_mapping = input_mapping or {}

    @property
    def name(self) -> str:
        """Get the tool name."""
        return self._tool_instance.name

    async def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Execute the decorated function.

        Args:
            input: Dictionary of function arguments
            config: Optional configuration

        Returns:
            Dictionary with execution result
        """
        # Apply input mapping
        mapped_input = {}
        for key, value in input.items():
            mapped_key = self._input_mapping.get(key, key)
            mapped_input[mapped_key] = value

        # Build execution context
        exec_ctx: Dict[str, Any] = {}
        if config:
            exec_ctx["tags"] = config.tags
            exec_ctx["metadata"] = config.metadata

        try:
            # Execute the tool instance's execute method
            result = await self._tool_instance.execute(exec_ctx, **mapped_input)

            output = {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "metadata": result.metadata or {},
                "tool_name": self._tool_instance.name,
            }

            if self._output_key and isinstance(result.output, dict):
                output["extracted"] = result.output.get(self._output_key)

            return output

        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "metadata": {"exception": type(e).__name__},
                "tool_name": self.name,
            }

    def __repr__(self) -> str:
        return f"FunctionToolRunnable({self.name})"


def as_runnable(
    tool_or_func: Union["BaseTool", Callable],
    output_key: Optional[str] = None,
    input_mapping: Optional[Dict[str, str]] = None,
    **bound_args: Any,
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    """Convert a tool or function to a Runnable.

    This is the primary entry point for using Victor tools in LCEL chains.

    Args:
        tool_or_func: A BaseTool instance or @tool decorated function
        output_key: Key to extract from output dict
        input_mapping: Map input keys to parameter names
        **bound_args: Arguments to bind to the runnable

    Returns:
        A Runnable wrapping the tool/function

    Example:
        from victor.tools.filesystem import read, ls

        # Create runnables
        read_runnable = as_runnable(read)
        ls_runnable = as_runnable(ls)

        # Build chain
        chain = ls_runnable | (lambda x: {"path": x["output"]["items"][0]["path"]}) | read_runnable

        # With bound args
        config_reader = as_runnable(read, path="config.yaml")
    """
    # Import here to avoid circular imports
    from victor.tools.base import BaseTool

    runnable: Runnable

    if isinstance(tool_or_func, BaseTool):
        runnable = ToolRunnable(tool_or_func, output_key, input_mapping)
    elif hasattr(tool_or_func, "Tool"):
        runnable = FunctionToolRunnable(tool_or_func, output_key, input_mapping)
    elif callable(tool_or_func):
        runnable = RunnableLambda(tool_or_func)
    else:
        raise TypeError(f"Cannot convert {type(tool_or_func)} to Runnable")

    if bound_args:
        runnable = runnable.bind(**bound_args)

    return runnable


# =============================================================================
# Chain Building Helpers
# =============================================================================


def chain(*runnables: Union[Runnable, Callable]) -> RunnableSequence:
    """Build a chain from multiple runnables.

    Alternative to chained `|` operators.

    Args:
        *runnables: Runnables or callables to chain

    Returns:
        A RunnableSequence

    Example:
        chain(read_tool, analyze, format)
    """
    converted: List[Runnable] = []
    for r in runnables:
        if isinstance(r, Runnable):
            converted.append(r)
        elif callable(r):
            converted.append(RunnableLambda(r))
        else:
            raise TypeError(f"Cannot chain {type(r)}")
    return RunnableSequence(converted)


def parallel(**steps: Union[Runnable, Callable]) -> RunnableParallel:
    """Build a parallel execution from named runnables.

    Convenience wrapper for RunnableParallel.

    Args:
        **steps: Named runnables or callables

    Returns:
        A RunnableParallel

    Example:
        parallel(
            summary=analyze_tool,
            security=security_scan,
        )
    """
    converted: Dict[str, Runnable] = {}
    for name, r in steps.items():
        if isinstance(r, Runnable):
            converted[name] = r
        elif callable(r):
            converted[name] = RunnableLambda(r)
        else:
            raise TypeError(f"Cannot parallelize {type(r)}")
    return RunnableParallel(converted)


def branch(
    *conditions: Tuple[Callable, Union[Runnable, Callable]],
    default: Optional[Union[Runnable, Callable]] = None,
) -> RunnableBranch:
    """Build a conditional branch from condition-runnable pairs.

    Convenience wrapper for RunnableBranch.

    Args:
        *conditions: Tuples of (condition_func, runnable_or_callable)
        default: Default runnable if no condition matches

    Returns:
        A RunnableBranch

    Example:
        branch(
            (is_python, python_lint),
            (is_javascript, js_lint),
            default=generic_check,
        )
    """
    branches: List[Tuple[Callable, Runnable]] = []
    for cond, r in conditions:
        if isinstance(r, Runnable):
            branches.append((cond, r))
        elif callable(r):
            branches.append((cond, RunnableLambda(r)))
        else:
            raise TypeError(f"Cannot branch to {type(r)}")

    default_runnable: Optional[Runnable] = None
    if default is not None:
        if isinstance(default, Runnable):
            default_runnable = default
        elif callable(default):
            default_runnable = RunnableLambda(default)
        else:
            raise TypeError(f"Cannot use {type(default)} as default")

    return RunnableBranch(*branches, default=default_runnable)


# =============================================================================
# Result Extractors (for chain transformations)
# =============================================================================


def extract_output(result: Dict[str, Any]) -> Any:
    """Extract the output field from a tool result.

    Useful in chains to get just the output value.

    Args:
        result: Tool result dictionary

    Returns:
        The output value
    """
    return result.get("output")


def extract_if_success(result: Dict[str, Any]) -> Any:
    """Extract output only if the tool succeeded.

    Args:
        result: Tool result dictionary

    Returns:
        Output if success, raises if failed

    Raises:
        RuntimeError: If tool execution failed
    """
    if not result.get("success", False):
        error = result.get("error", "Unknown error")
        raise RuntimeError(f"Tool failed: {error}")
    return result.get("output")


def map_keys(mapping: Dict[str, str]) -> Callable[[Dict], Dict]:
    """Create a function that renames dictionary keys.

    Useful for adapting output of one tool to input of another.

    Args:
        mapping: Dict of old_key -> new_key

    Returns:
        A function that applies the mapping

    Example:
        chain = tool1 | map_keys({"result": "input"}) | tool2
    """

    def mapper(d: Dict) -> Dict:
        result = dict(d)
        for old, new in mapping.items():
            if old in result:
                result[new] = result.pop(old)
        return result

    return mapper


def select_keys(*keys: str) -> Callable[[Dict], Dict]:
    """Create a function that selects specific keys from a dict.

    Args:
        *keys: Keys to keep

    Returns:
        A function that filters to specified keys
    """

    def selector(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if k in keys}

    return selector


__all__ = [
    # Core classes
    "Runnable",
    "RunnableConfig",
    "RunnableSequence",
    "RunnableParallel",
    "RunnableBranch",
    "RunnableLambda",
    "RunnablePassthrough",
    "RunnableBinding",
    # Tool adapters
    "ToolRunnable",
    "FunctionToolRunnable",
    "as_runnable",
    # Chain builders
    "chain",
    "parallel",
    "branch",
    # Extractors
    "extract_output",
    "extract_if_success",
    "map_keys",
    "select_keys",
]
