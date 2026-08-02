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

"""Compute node executor.

Executes compute nodes by calling registered handler functions.
This is a stub that delegates to legacy implementation during migration.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.workflows.definition import ComputeNode
    from victor.workflows.runtime_types import WorkflowState

logger = logging.getLogger(__name__)


class ComputeNodeExecutor:
    """Executor for compute nodes.

    Responsibility (SRP):
    - Execute registered handler functions
    - Map handler names to handler instances
    - Pass input context to handlers
    - Handle handler errors

    Non-responsibility:
    - Workflow compilation (handled by WorkflowCompiler)
    - Handler registration (handled by ComputeHandlerRegistry)
    """

    def __init__(self, context: Any = None):
        """Initialize the executor.

        Args:
            context: ExecutionContext with services, settings
        """
        self._context = context

    async def execute(self, node: "ComputeNode", state: "WorkflowState") -> "WorkflowState":
        """Execute a compute node.

        Args:
            node: Compute node definition
            state: Current workflow state

        Returns:
            Updated workflow state

        Raises:
            Exception: If handler execution fails
        """
        import asyncio
        from victor.workflows.runtime_types import GraphNodeResult

        logger.info(f"Executing compute node: {node.id}")
        start_time = time.time()

        # Step 1: Build params from node input_mapping with $ctx. and $state. prefixes
        params = {}
        if node.input_mapping:
            for param_name, source in node.input_mapping.items():
                # Handle $ctx. prefix (from state)
                if isinstance(source, str) and source.startswith("$ctx."):
                    context_key = source[5:]
                    if context_key in state:
                        params[param_name] = state[context_key]
                    else:
                        params[param_name] = context_key
                # Handle $state. prefix (also from state)
                elif isinstance(source, str) and source.startswith("$state."):
                    context_key = source[7:]
                    if context_key in state:
                        params[param_name] = state[context_key]
                    else:
                        params[param_name] = context_key
                # Direct value
                else:
                    params[param_name] = source

        # Step 2: Get tool registry from context
        tool_registry = None
        if self._context and hasattr(self._context, "tool_registry"):
            tool_registry = self._context.tool_registry
        elif (
            self._context
            and hasattr(self._context, "services")
            and self._context.services is not None
        ):
            from victor.tools.registry import ToolRegistry

            services = self._context.services
            if hasattr(services, "get_optional"):
                tool_registry = services.get_optional(ToolRegistry)

        if tool_registry is None:
            try:
                from victor.tools.registry import get_tool_registry

                tool_registry = get_tool_registry()
            except Exception:
                tool_registry = None

        tool_calls_used = 0
        output = None

        # Step 3: Check for custom handler
        if node.handler:
            handler = self._get_compute_handler(node.handler)
            if handler:
                # Create minimal WorkflowContext wrapper
                from victor.workflows.context import WorkflowContext

                context = WorkflowContext(dict(state))
                result = await handler(node, context, tool_registry)

                # Transfer context changes back to state
                for key, value in context.data.items():
                    if not key.startswith("_"):
                        state[key] = value

                output = result.output if result else None
                tool_calls_used = result.tool_calls_used if result else 0
            else:
                logger.warning(f"Handler '{node.handler}' not found for node '{node.id}'")
                output = {"error": f"Handler '{node.handler}' not found"}
        else:
            # Step 4: Execute tools directly
            outputs = {}
            if tool_registry and node.tools:
                for tool_name in node.tools:
                    # Check constraints
                    if hasattr(node, "constraints") and node.constraints:
                        if not node.constraints.allows_tool(tool_name):
                            logger.debug(f"Tool '{tool_name}' blocked by constraints")
                            continue

                    try:
                        timeout = getattr(node, "timeout", 300)
                        if hasattr(node, "constraints") and node.constraints:
                            timeout = getattr(node.constraints, "timeout", 300)

                        result = await asyncio.wait_for(
                            tool_registry.execute(
                                tool_name,
                                _exec_ctx={
                                    "workflow_context": state,
                                    "constraints": (
                                        node.constraints.to_dict()
                                        if hasattr(node, "constraints") and node.constraints
                                        else {}
                                    ),
                                },
                                **params,
                            ),
                            timeout=timeout,
                        )
                        tool_calls_used += 1

                        if result.success:
                            outputs[tool_name] = result.output
                        else:
                            outputs[tool_name] = {"error": result.error}

                        if hasattr(node, "fail_fast") and node.fail_fast and not result.success:
                            break

                    except asyncio.TimeoutError:
                        outputs[tool_name] = {"error": "Timeout"}
                        if hasattr(node, "fail_fast") and node.fail_fast:
                            break
                    except Exception as e:
                        logger.error(f"Tool '{tool_name}' execution failed: {e}")
                        outputs[tool_name] = {"error": str(e)}
                        if hasattr(node, "fail_fast") and node.fail_fast:
                            break
            else:
                outputs = {"status": "no_tools_executed", "params": params}

            output = outputs

        # Step 5: Store output in state
        output_key = node.output_key or node.id
        state[output_key] = output

        # Step 6: Update node results for observability
        if "_node_results" not in state:
            state["_node_results"] = {}

        state["_node_results"][node.id] = GraphNodeResult(
            node_id=node.id,
            success=True,
            output=output,
            duration_seconds=time.time() - start_time,
            tool_calls_used=tool_calls_used,
        )

        logger.info(f"Compute node {node.id} completed successfully")
        return state

    def _get_compute_handler(self, handler_name: str) -> Any:
        """Get compute handler by name, including chain: prefix resolution.

        Args:
            handler_name: Name of the registered handler (e.g., "my_handler" or "chain:vertical:name")

        Returns:
            Handler instance or None if not found
        """
        from victor.workflows.compute_registry import get_compute_handler
        from victor.workflows.compute_registry import CHAIN_HANDLER_PREFIX

        # Check if this is a chain reference (e.g., "chain:coding:static_analysis")
        if handler_name.startswith(CHAIN_HANDLER_PREFIX):
            return self._resolve_chain_handler(handler_name)

        return get_compute_handler(handler_name)

    def _resolve_chain_handler(self, handler_name: str) -> Any:
        """Resolve a chain:vertical:name reference to a ComputeHandler.

        Implements the Agent-as-Tool protocol from research:
        - Parses "chain:vertical:name" or "chain:name"
        - Creates chain from registry (lazy evaluation)
        - Wraps Runnable as ComputeHandler with standardized protocol

        Args:
            handler_name: Full handler string with "chain:" prefix

        Returns:
            ComputeHandler wrapper or None if chain not found
        """
        from victor.workflows.compute_registry import CHAIN_HANDLER_PREFIX
        from victor.framework.chain_registry import create_chain

        # Parse "chain:vertical:name" or "chain:name"
        chain_ref = handler_name[len(CHAIN_HANDLER_PREFIX) :]
        if ":" in chain_ref:
            vertical, name = chain_ref.split(":", 1)
        else:
            vertical, name = None, chain_ref

        # Create chain from registry (lazy evaluation)
        runnable = create_chain(name, vertical=vertical)
        if runnable is None:
            logger.warning(
                f"Chain '{chain_ref}' not found in registry " f"(vertical={vertical}, name={name})"
            )
            return None

        logger.info(f"Resolved chain handler '{chain_ref}' to runnable")
        return self._create_chain_wrapper(runnable, chain_ref)

    def _create_chain_wrapper(self, runnable: Any, chain_ref: str) -> Any:
        """Wrap a Runnable as a ComputeHandler following Agent-as-Tool protocol.

        Implements the protocol: Tk(p) = (v, σ)
        - p: structured parameter object (input_data)
        - v: structured result (output)
        - σ: execution status (NodeResult.status)

        Args:
            runnable: The Runnable chain to wrap
            chain_ref: Original chain reference for logging

        Returns:
            Async ComputeHandler function
        """
        import asyncio
        from victor_contracts.workflows import ExecutorNodeStatus, NodeResult

        async def chain_handler(
            node: "ComputeNode",
            context: Any,
            tool_registry: Any,
        ) -> "NodeResult":
            """Execute the chain as a compute handler."""
            # Prepare input data from context and input_mapping
            input_data = self._prepare_chain_input(node, context)

            logger.debug(f"Executing chain '{chain_ref}' with input: {list(input_data.keys())}")

            # Get timeout from node
            timeout = getattr(node, "timeout", 300)

            try:
                # Execute chain with timeout
                output = await asyncio.wait_for(runnable.invoke(input_data), timeout=timeout)

                # Update context with output
                self._update_context(context, node, output)

                logger.info(f"Chain '{chain_ref}' completed successfully")

                return NodeResult(
                    node_id=node.id, status=ExecutorNodeStatus.COMPLETED, output=output
                )

            except asyncio.TimeoutError:
                error = f"Chain '{chain_ref}' timed out after {timeout}s"
                logger.error(error)
                return NodeResult(node_id=node.id, status=ExecutorNodeStatus.FAILED, error=error)

            except Exception as e:
                error = f"Chain '{chain_ref}' execution failed: {e}"
                logger.error(error, exc_info=True)
                return NodeResult(node_id=node.id, status=ExecutorNodeStatus.FAILED, error=error)

        return chain_handler

    def _prepare_chain_input(self, node: "ComputeNode", context: Any) -> dict:
        """Prepare input data for chain invocation from context and input_mapping.

        Args:
            node: Compute node with optional input_mapping
            context: Workflow context

        Returns:
            Dictionary of input data for the chain
        """
        input_data = {}

        # Use input_mapping if provided
        if hasattr(node, "input_mapping") and node.input_mapping:
            for key, source in node.input_mapping.items():
                # Handle $ctx. prefix
                if isinstance(source, str) and source.startswith("$ctx."):
                    context_key = source[5:]
                    input_data[key] = context.get(context_key, context_key)
                # Handle $state. prefix
                elif isinstance(source, str) and source.startswith("$state."):
                    context_key = source[7:]
                    input_data[key] = context.get(context_key, context_key)
                # Direct value
                else:
                    input_data[key] = source
        else:
            # Fall back to all context data
            if hasattr(context, "data"):
                input_data = dict(context.data)
            elif isinstance(context, dict):
                input_data = dict(context)

        return input_data

    def _update_context(self, context: Any, node: "ComputeNode", output: Any) -> None:
        """Update workflow context with chain output.

        Args:
            context: Workflow context to update
            node: Compute node with optional output_key
            output: Chain execution output
        """
        # Use output_key if specified
        if hasattr(node, "output_key") and node.output_key:
            context.set(node.output_key, output)
        # Otherwise, merge dict output
        elif isinstance(output, dict):
            for key, value in output.items():
                if not key.startswith("_"):
                    context.set(key, value)

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this executor supports the given node type."""
        return node_type == "compute"


__all__ = ["ComputeNodeExecutor"]
