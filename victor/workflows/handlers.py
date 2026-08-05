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

"""Framework-level compute handlers for common workflow patterns.

Provides reusable handlers that can be used across all verticals:
- parallel_tools: Execute tools concurrently with semaphore control
- sequential_tools: Chain tools with output piping
- retry_with_backoff: Retry failed tools with exponential backoff
- data_transform: Apply transformations to context data
- conditional_branch: Evaluate conditions and determine next node

Example YAML usage:
    - id: parallel_fetch
      type: compute
      handler: parallel_tools
      tools: [fetch_data, fetch_metadata]
      output: combined_data

    - id: transform
      type: compute
      handler: data_transform
      inputs:
        data: $ctx.raw_data
        operations: [normalize, fill_missing]
      output: clean_data
"""

from __future__ import annotations

import ast
import asyncio
import logging
import operator
import time
import traceback
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor_contracts.workflows import ExecutorNodeStatus, NodeResult

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.context import WorkflowContext

logger = logging.getLogger(__name__)


# =============================================================================
# Error Boundary for Compute Handlers
# =============================================================================


@dataclass
class HandlerError:
    """Detailed error information from handler execution.

    Captures full context when a compute handler fails, enabling
    better debugging and error classification for all verticals.

    Attributes:
        handler_name: Name of the handler that failed
        node_id: ID of the workflow node
        error_type: Classification of the error (timeout, validation, etc.)
        message: Human-readable error message
        traceback_str: Full traceback as string (for debugging)
        context_snapshot: Context data at time of failure
    """

    handler_name: str
    node_id: str
    error_type: str
    message: str
    traceback_str: Optional[str] = None
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "handler_name": self.handler_name,
            "node_id": self.node_id,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback_str,
        }


class HandlerErrorBoundary:
    """Error boundary wrapper for compute handlers.

    Provides exception isolation, context preservation, and structured
    error reporting for compute handler execution. All verticals
    (including third-party plugins) benefit from consistent error handling.

    Features:
    - Exception isolation per handler
    - Context state preservation on error
    - Detailed error classification (timeout, validation, runtime)
    - Structured error reporting with HandlerError

    Example:
        boundary = HandlerErrorBoundary()
        result = await boundary.execute(
            handler=my_handler,
            handler_name="custom_compute",
            node=compute_node,
            context=ctx,
            tool_registry=registry,
        )

        # With decorator
        @with_error_boundary("my_handler")
        async def my_handler(node, context, tool_registry):
            ...
    """

    def __init__(self, preserve_context: bool = True):
        """Initialize error boundary.

        Args:
            preserve_context: If True, snapshot context before execution
        """
        self.preserve_context = preserve_context

    async def execute(
        self,
        handler: Callable,
        handler_name: str,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        """Execute handler with error boundary protection.

        Args:
            handler: The compute handler to execute
            handler_name: Name for error reporting
            node: The ComputeNode being executed
            context: Workflow execution context
            tool_registry: Tool registry for tool execution

        Returns:
            NodeResult with execution outcome or error details
        """
        start_time = time.time()

        # Snapshot context before execution (for debugging on failure)
        context_snapshot = {}
        if self.preserve_context:
            try:
                # Only snapshot primitive values to avoid issues
                context_snapshot = {
                    k: v
                    for k, v in context.data.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
            except Exception:
                pass

        try:
            result = await handler(node, context, tool_registry)
            return result

        except asyncio.TimeoutError as e:
            error = HandlerError(
                handler_name=handler_name,
                node_id=node.id,
                error_type="timeout",
                message=str(e) or "Handler execution timed out",
                context_snapshot=context_snapshot,
            )
            logger.error(f"Handler timeout: {handler_name} on node {node.id}")
            return self._create_error_result(node, error, start_time)

        except ValueError as e:
            error = HandlerError(
                handler_name=handler_name,
                node_id=node.id,
                error_type="validation",
                message=str(e),
                traceback_str=traceback.format_exc(),
                context_snapshot=context_snapshot,
            )
            logger.error(f"Handler validation error: {handler_name}: {e}")
            return self._create_error_result(node, error, start_time)

        except KeyError as e:
            error = HandlerError(
                handler_name=handler_name,
                node_id=node.id,
                error_type="missing_key",
                message=f"Missing required key: {e}",
                traceback_str=traceback.format_exc(),
                context_snapshot=context_snapshot,
            )
            logger.error(f"Handler missing key: {handler_name}: {e}")
            return self._create_error_result(node, error, start_time)

        except TypeError as e:
            error = HandlerError(
                handler_name=handler_name,
                node_id=node.id,
                error_type="type_error",
                message=str(e),
                traceback_str=traceback.format_exc(),
                context_snapshot=context_snapshot,
            )
            logger.error(f"Handler type error: {handler_name}: {e}")
            return self._create_error_result(node, error, start_time)

        except Exception as e:
            error = HandlerError(
                handler_name=handler_name,
                node_id=node.id,
                error_type=type(e).__name__,
                message=str(e),
                traceback_str=traceback.format_exc(),
                context_snapshot=context_snapshot,
            )
            logger.error(
                f"Handler exception: {handler_name} on node {node.id}: {e}",
                exc_info=True,
            )
            return self._create_error_result(node, error, start_time)

    def _create_error_result(
        self,
        node: "ComputeNode",
        error: HandlerError,
        start_time: float,
    ) -> "NodeResult":
        """Create NodeResult from HandlerError.

        Args:
            node: The failed node
            error: Error details
            start_time: When execution started

        Returns:
            NodeResult with FAILED status and error details
        """
        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.FAILED,
            error=f"Handler '{error.handler_name}' failed: {error.message}",
            duration_seconds=time.time() - start_time,
        )


def with_error_boundary(handler_name: str):
    """Decorator to wrap handler with error boundary.

    Provides automatic error boundary wrapping for compute handlers,
    ensuring consistent error handling across all verticals.

    Args:
        handler_name: Name for error reporting

    Example:
        @with_error_boundary("my_custom_handler")
        async def my_handler(node, context, tool_registry):
            # Handler implementation
            ...

        # Register with executor
        register_compute_handler("my_custom", my_handler)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(
            node: "ComputeNode",
            context: "WorkflowContext",
            tool_registry: "ToolRegistry",
        ) -> "NodeResult":
            boundary = HandlerErrorBoundary()
            return await boundary.execute(
                handler=func,
                handler_name=handler_name,
                node=node,
                context=context,
                tool_registry=tool_registry,
            )

        return wrapper

    return decorator


@dataclass
class ParallelToolsHandler:
    """Execute tools in parallel with semaphore control.

    This handler runs all tools specified in the node concurrently,
    respecting the max_concurrent limit via semaphore. Results are
    aggregated into a dictionary keyed by tool name.

    Attributes:
        max_concurrent: Maximum number of tools to execute simultaneously

    Example:
        register_compute_handler("parallel_tools", ParallelToolsHandler(max_concurrent=4))

        # In YAML:
        - id: fetch_all
          type: compute
          handler: parallel_tools
          tools: [fetch_sec, fetch_prices, fetch_news]
          output: all_data
    """

    max_concurrent: int = 4

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        outputs: Dict[str, Any] = {}
        errors: List[str] = []
        tool_calls_used = 0

        # Build params from input mapping
        params = self._build_params(node, context)

        async def execute_tool(tool_name: str) -> tuple:
            nonlocal tool_calls_used
            async with semaphore:
                try:
                    # Check constraints before execution
                    if not node.constraints.allows_tool(tool_name):
                        return (
                            tool_name,
                            None,
                            f"Tool '{tool_name}' blocked by constraints",
                        )

                    result = await asyncio.wait_for(
                        tool_registry.execute(
                            tool_name,
                            _exec_ctx={
                                "workflow_context": context.data,
                                "constraints": node.constraints.to_dict(),
                            },
                            **params,
                        ),
                        timeout=node.constraints.timeout,
                    )
                    tool_calls_used += 1

                    if result.success:
                        return tool_name, result.output, None
                    else:
                        return tool_name, None, result.error

                except asyncio.TimeoutError:
                    return (
                        tool_name,
                        None,
                        f"Timed out after {node.constraints.timeout}s",
                    )
                except Exception as e:
                    return tool_name, None, str(e)

        # Execute all tools in parallel
        tasks = [execute_tool(tool_name) for tool_name in node.tools]
        results = await asyncio.gather(*tasks)

        for tool_name, output, error in results:
            if error:
                errors.append(f"{tool_name}: {error}")
                if node.fail_fast:
                    break
            elif output is not None:
                outputs[tool_name] = output

        # Store in context
        if node.output_key:
            context.set(node.output_key, outputs)

        # Determine status
        if errors and node.fail_fast:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                output=outputs,
                error="; ".join(errors),
                duration_seconds=time.time() - start_time,
                tool_calls_used=tool_calls_used,
            )

        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=outputs,
            error="; ".join(errors) if errors else None,
            duration_seconds=time.time() - start_time,
            tool_calls_used=tool_calls_used,
        )

    def _build_params(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
    ) -> Dict[str, Any]:
        """Build tool parameters from context."""
        params = {}
        for param_name, context_key in node.input_mapping.items():
            value = context.get(context_key)
            if value is not None:
                params[param_name] = value
            else:
                params[param_name] = context_key
        return params


@dataclass
class SequentialToolsHandler:
    """Execute tools sequentially with output chaining.

    This handler runs tools one after another, optionally passing
    the output of each tool as input to the next. Useful for
    pipelines where each step depends on the previous.

    Attributes:
        chain_outputs: If True, pass previous tool's output to next
        stop_on_error: If True, stop execution on first error

    Example:
        register_compute_handler("sequential_tools", SequentialToolsHandler())

        # In YAML:
        - id: etl_pipeline
          type: compute
          handler: sequential_tools
          tools: [extract, transform, load]
          output: loaded_data
    """

    chain_outputs: bool = True
    stop_on_error: bool = True

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        outputs: Dict[str, Any] = {}
        tool_calls_used = 0
        params = self._build_params(node, context)
        last_output = None

        for tool_name in node.tools:
            # Check constraints
            if not node.constraints.allows_tool(tool_name):
                if self.stop_on_error:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        output=outputs,
                        error=f"Tool '{tool_name}' blocked by constraints",
                        duration_seconds=time.time() - start_time,
                        tool_calls_used=tool_calls_used,
                    )
                continue

            # Chain output from previous tool
            tool_params = dict(params)
            if self.chain_outputs and last_output is not None:
                tool_params["_previous_output"] = last_output

            try:
                result = await asyncio.wait_for(
                    tool_registry.execute(
                        tool_name,
                        _exec_ctx={
                            "workflow_context": context.data,
                            "constraints": node.constraints.to_dict(),
                        },
                        **tool_params,
                    ),
                    timeout=node.constraints.timeout,
                )
                tool_calls_used += 1

                if result.success:
                    outputs[tool_name] = result.output
                    last_output = result.output
                elif self.stop_on_error:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        output=outputs,
                        error=f"Tool '{tool_name}' failed: {result.error}",
                        duration_seconds=time.time() - start_time,
                        tool_calls_used=tool_calls_used,
                    )

            except asyncio.TimeoutError:
                if self.stop_on_error:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        output=outputs,
                        error=f"Tool '{tool_name}' timed out",
                        duration_seconds=time.time() - start_time,
                        tool_calls_used=tool_calls_used,
                    )

            except Exception as e:
                if self.stop_on_error:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        output=outputs,
                        error=f"Tool '{tool_name}' error: {e}",
                        duration_seconds=time.time() - start_time,
                        tool_calls_used=tool_calls_used,
                    )

        # Store final output
        if node.output_key:
            context.set(node.output_key, last_output if self.chain_outputs else outputs)

        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=outputs,
            duration_seconds=time.time() - start_time,
            tool_calls_used=tool_calls_used,
        )

    def _build_params(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
    ) -> Dict[str, Any]:
        """Build tool parameters from context."""
        params = {}
        for param_name, context_key in node.input_mapping.items():
            value = context.get(context_key)
            if value is not None:
                params[param_name] = value
            else:
                params[param_name] = context_key
        return params


@dataclass
class RetryBackoffHandler:
    """Retry failed tools with exponential backoff.

    This handler wraps tool execution with retry logic, using
    exponential backoff between attempts. Useful for handling
    transient failures in network calls or external services.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Multiplier for exponential backoff

    Example:
        register_compute_handler("retry_with_backoff", RetryBackoffHandler())

        # In YAML:
        - id: fetch_api
          type: compute
          handler: retry_with_backoff
          tools: [call_external_api]
          output: api_response
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        outputs: Dict[str, Any] = {}
        tool_calls_used = 0
        params = self._build_params(node, context)

        for tool_name in node.tools:
            if not node.constraints.allows_tool(tool_name):
                continue

            last_error = None
            success = False

            for attempt in range(self.max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        tool_registry.execute(
                            tool_name,
                            _exec_ctx={
                                "workflow_context": context.data,
                                "constraints": node.constraints.to_dict(),
                            },
                            **params,
                        ),
                        timeout=node.constraints.timeout,
                    )
                    tool_calls_used += 1

                    if result.success:
                        outputs[tool_name] = result.output
                        success = True
                        break
                    else:
                        last_error = result.error

                except asyncio.TimeoutError:
                    last_error = f"Timed out after {node.constraints.timeout}s"

                except Exception as e:
                    last_error = str(e)

                # Calculate backoff delay
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base**attempt),
                        self.max_delay,
                    )
                    logger.debug(
                        f"Retry {attempt + 1}/{self.max_retries} for {tool_name} "
                        f"after {delay:.1f}s delay"
                    )
                    await asyncio.sleep(delay)

            if not success and node.fail_fast:
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    output=outputs,
                    error=f"Tool '{tool_name}' failed after {self.max_retries} retries: {last_error}",
                    duration_seconds=time.time() - start_time,
                    tool_calls_used=tool_calls_used,
                )

        # Store outputs
        if node.output_key:
            context.set(node.output_key, outputs)

        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=outputs,
            duration_seconds=time.time() - start_time,
            tool_calls_used=tool_calls_used,
        )

    def _build_params(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
    ) -> Dict[str, Any]:
        """Build tool parameters from context."""
        params = {}
        for param_name, context_key in node.input_mapping.items():
            value = context.get(context_key)
            if value is not None:
                params[param_name] = value
            else:
                params[param_name] = context_key
        return params


@dataclass
class DataTransformHandler:
    """Apply transformations to context data.

    This handler applies a series of transformation functions
    to data from the context, without requiring tool execution.
    Useful for data preprocessing, normalization, and aggregation.

    The handler looks for 'operations' in the node's input_mapping
    and applies registered transform functions.

    Example:
        register_compute_handler("data_transform", DataTransformHandler())

        # In YAML:
        - id: clean_data
          type: compute
          handler: data_transform
          inputs:
            data: $ctx.raw_data
            operations: [normalize, fill_missing, deduplicate]
          output: clean_data
    """

    # Registry of transform functions
    _transforms: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Register default transforms
        self._transforms = {
            "normalize": self._normalize,
            "fill_missing": self._fill_missing,
            "deduplicate": self._deduplicate,
            "flatten": self._flatten,
            "to_list": self._to_list,
            "to_dict": self._to_dict,
            "filter_none": self._filter_none,
        }

    def register_transform(self, name: str, func: Callable[[Any], Any]) -> None:
        """Register a custom transform function."""
        self._transforms[name] = func

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        # Get data and operations from inputs
        data = None
        operations: List[str] = []

        for param_name, context_key in node.input_mapping.items():
            value = context.get(context_key)
            if param_name == "data":
                data = value if value is not None else context_key
            elif param_name == "operations":
                operations = value if isinstance(value, list) else [value]

        if data is None:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error="No 'data' input provided for transform",
                duration_seconds=time.time() - start_time,
            )

        # Apply operations
        result = data
        for op_name in operations:
            if op_name in self._transforms:
                try:
                    result = self._transforms[op_name](result)
                except Exception as e:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        error=f"Transform '{op_name}' failed: {e}",
                        duration_seconds=time.time() - start_time,
                    )
            else:
                logger.warning(f"Unknown transform: {op_name}")

        # Store result
        if node.output_key:
            context.set(node.output_key, result)

        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=result,
            duration_seconds=time.time() - start_time,
        )

    # Default transform functions
    def _normalize(self, data: Any) -> Any:
        """Normalize data (placeholder - extend for specific types)."""
        if isinstance(data, list):
            return [self._normalize(item) for item in data]
        if isinstance(data, dict):
            return {k: self._normalize(v) for k, v in data.items()}
        if isinstance(data, str):
            return data.strip().lower()
        return data

    def _fill_missing(self, data: Any) -> Any:
        """Fill missing values (placeholder)."""
        if isinstance(data, dict):
            return {k: v if v is not None else "" for k, v in data.items()}
        return data

    def _deduplicate(self, data: Any) -> Any:
        """Remove duplicates from lists."""
        if isinstance(data, list):
            seen = set()
            result = []
            for item in data:
                key = str(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result
        return data

    def _flatten(self, data: Any) -> Any:
        """Flatten nested lists."""
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, list):
                    result.extend(self._flatten(item))
                else:
                    result.append(item)
            return result
        return data

    def _to_list(self, data: Any) -> list:
        """Convert to list."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
        return [data]

    def _to_dict(self, data: Any) -> dict:
        """Convert to dict."""
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return dict(enumerate(data))
        return {"value": data}

    def _filter_none(self, data: Any) -> Any:
        """Remove None values."""
        if isinstance(data, list):
            return [x for x in data if x is not None]
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


@dataclass
class ConditionalBranchHandler:
    """Evaluate conditions and determine next node.

    This handler evaluates a condition expression against the
    context data and returns a result indicating which branch
    to take. The condition is specified in the node's input_mapping.

    Supports simple comparison operators and context variable references.

    Example:
        register_compute_handler("conditional_branch", ConditionalBranchHandler())

        # In YAML:
        - id: check_quality
          type: compute
          handler: conditional_branch
          inputs:
            condition: "quality_score > 0.8"
          output: branch_decision
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        # Get condition from inputs
        condition_expr = None
        for param_name, value in node.input_mapping.items():
            if param_name == "condition":
                condition_expr = value
                break

        if not condition_expr:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error="No 'condition' input provided",
                duration_seconds=time.time() - start_time,
            )

        try:
            result = self._evaluate_condition(condition_expr, context.data)

            if node.output_key:
                context.set(node.output_key, result)

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output={"result": result, "condition": condition_expr},
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=f"Condition evaluation failed: {e}",
                duration_seconds=time.time() - start_time,
            )

    def _evaluate_condition(self, expr: str, context_data: Dict[str, Any]) -> bool:
        """Evaluate a simple condition expression.

        Supports:
        - Comparison: >, <, >=, <=, ==, !=
        - Boolean: and, or, not
        - Context variables: ${var_name} or just var_name

        The expression is parsed and walked as a restricted AST. Only literals,
        boolean/comparison/unary operators and names resolved from
        ``context_data`` are permitted — there is no call, attribute, or
        subscript access, so arbitrary code execution is not possible.
        """
        import re

        # Normalize ${var} / $var references to bare identifiers; the actual
        # values are supplied via the names map below, never string-substituted.
        normalized = re.sub(r"\$\{(\w+)\}", r"\1", expr)
        normalized = re.sub(r"\$(\w+)", r"\1", normalized)

        names: Dict[str, Any] = {"True": True, "False": False, "None": None}
        names.update(context_data)

        try:
            tree = ast.parse(normalized, mode="eval")
            return bool(_safe_eval_node(tree.body, names))
        except Exception as e:
            logger.warning(f"Condition eval failed: {expr} -> {normalized}: {e}")
            return False


# Restricted operators permitted in workflow condition expressions. No call,
# attribute, subscript, or name binding is allowed by _safe_eval_node, so these
# expressions cannot execute arbitrary code (unlike the eval() they replace).
_SAFE_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_SAFE_UNARY_OPS = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST, names: Dict[str, Any]) -> Any:
    """Evaluate a restricted AST node for workflow condition expressions.

    Only literals, names resolved from ``names``, boolean operators,
    comparisons and unary operators are supported. Any other node type raises
    ``ValueError`` so untrusted expressions cannot reach dangerous constructs.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(f"Unknown name in condition: {node.id}")
    if isinstance(node, ast.BoolOp):
        values = [_safe_eval_node(v, names) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_safe_eval_node(node.operand, names))
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _SAFE_COMPARE_OPS:
                raise ValueError(f"Unsupported comparison: {op_type.__name__}")
            right = _safe_eval_node(comparator, names)
            if not _SAFE_COMPARE_OPS[op_type](left, right):
                return False
            left = right
        return True
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


# Framework handler instances
FRAMEWORK_HANDLERS: Dict[str, Any] = {
    "parallel_tools": ParallelToolsHandler(),
    "sequential_tools": SequentialToolsHandler(),
    "retry_with_backoff": RetryBackoffHandler(),
    "data_transform": DataTransformHandler(),
    "conditional_branch": ConditionalBranchHandler(),
}


def register_framework_handlers() -> None:
    """Register all framework-level handlers with the executor.

    Call this function during application initialization to make
    framework handlers available for use in YAML workflows.

    Example:
        from victor.workflows.handlers import register_framework_handlers
        register_framework_handlers()
    """
    from victor.workflows.compute_registry import register_compute_handler

    for name, handler in FRAMEWORK_HANDLERS.items():
        register_compute_handler(name, handler)
        logger.debug(f"Registered framework handler: {name}")


def get_framework_handler(name: str) -> Optional[Any]:
    """Get a framework handler by name."""
    return FRAMEWORK_HANDLERS.get(name)


def list_framework_handlers() -> List[str]:
    """List all available framework handler names."""
    return list(FRAMEWORK_HANDLERS.keys())


__all__ = [
    # Error boundary
    "HandlerError",
    "HandlerErrorBoundary",
    "with_error_boundary",
    # Handler classes
    "ParallelToolsHandler",
    "SequentialToolsHandler",
    "RetryBackoffHandler",
    "DataTransformHandler",
    "ConditionalBranchHandler",
    # Registration
    "FRAMEWORK_HANDLERS",
    "register_framework_handlers",
    "get_framework_handler",
    "list_framework_handlers",
]
