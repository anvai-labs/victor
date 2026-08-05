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

"""Delegate tool for agent-to-agent task delegation.

This module provides DelegateTool, which allows agents to spawn specialized
sub-agents for specific tasks during execution. This enables:

- Dynamic task decomposition at runtime
- Parallel work on independent sub-problems
- Specialized agents for research, planning, execution, etc.
- Fire-and-forget background tasks

The tool integrates with Victor's tool system and appears as a regular
tool that LLMs can call.

Example (from agent's perspective):
    # The LLM can call the delegate tool like any other tool
    result = delegate(
        task="Find all files that use the deprecated auth module",
        role="researcher",
        tool_budget=15,
    )
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.tools.base import BaseTool, CostTier, ToolResult
from victor.agent.delegation.protocol import DelegationRequest

if TYPE_CHECKING:
    from victor.agent.delegation.handler import DelegationHandler

logger = logging.getLogger(__name__)


class DelegateTool(BaseTool):
    """Tool that allows agents to delegate tasks to specialized sub-agents.

    This tool enables hierarchical agent structures where a main agent can
    spawn specialized sub-agents for specific tasks. Use cases include:

    - **Research delegation**: Spawn a researcher to explore the codebase
    - **Parallel exploration**: Delegate multiple independent searches
    - **Expert handoff**: Let a specialized agent handle complex sub-tasks
    - **Background work**: Fire-and-forget tasks that run in parallel

    The tool supports five specialized roles:
    - **researcher**: Read-only exploration (search, read, grep)
    - **planner**: Task breakdown and planning
    - **executor**: Code changes and execution
    - **reviewer**: Quality checks and validation
    - **tester**: Test writing and running

    Attributes:
        name: Tool name ("delegate")
        description: Tool description for LLM
        parameters: JSON Schema for parameters
        cost_tier: MEDIUM (spawning agents has compute cost)

    Example:
        # Agent calls delegate to research authentication code
        result = await delegate_tool.execute(
            task="Find all authentication endpoints and their handlers",
            role="researcher",
            tool_budget=15,
        )

        if result.success:
            print(result.result)  # Researcher's findings
    """

    name = "delegate"
    description = """Delegate a task to a specialized sub-agent. Use when:

- You need to explore or research a specific aspect while continuing other work
- The task requires specialized expertise (research, testing, review)
- You want parallel work on independent sub-problems
- The task would benefit from fresh context and focus

Available roles:
- researcher: Find information, explore code, search files (read-only)
- planner: Break down tasks, create plans, organize work
- executor: Make code changes, run commands, implement features
- reviewer: Check quality, validate changes, verify correctness
- tester: Write and run tests, check coverage

Returns the result from the delegated agent. By default, waits for completion."""

    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear description of what the delegate should accomplish. Be specific about the expected output.",
            },
            "role": {
                "type": "string",
                "enum": ["researcher", "planner", "executor", "reviewer", "tester"],
                "description": "Specialization for the delegate. researcher=read-only exploration, executor=code changes, etc.",
                "default": "executor",
            },
            "tool_budget": {
                "type": "integer",
                "description": "Maximum tool calls for the delegate (1-50). Higher for complex tasks.",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
            },
            "await_result": {
                "type": "boolean",
                "description": "Wait for completion (true) or fire-and-forget (false). Default true.",
                "default": True,
            },
            "context": {
                "type": "object",
                "description": "Additional context to pass to the delegate (key-value pairs).",
                "additionalProperties": True,
            },
        },
        "required": ["task"],
    }

    cost_tier = CostTier.MEDIUM  # Spawning agents has compute cost

    def __init__(
        self,
        handler: "DelegationHandler",
        parent_agent_id: str = "main",
        parent_goal: Optional[str] = None,
    ):
        """Initialize delegate tool.

        Args:
            handler: DelegationHandler for processing requests
            parent_agent_id: ID of the parent agent using this tool
            parent_goal: Optional context about parent's goal
        """
        super().__init__()
        self.handler = handler
        self.parent_agent_id = parent_agent_id
        self.parent_goal = parent_goal

    def sanitize_task(self, task: str) -> str:
        """Sanitize the task description to remove leaking thinking blocks.

        Args:
            task: Raw task string from the LLM

        Returns:
            Sanitized task string
        """
        # Remove common thinking blocks/tags
        # 1. <thinking>...</thinking> anywhere in the string
        task = re.sub(r"<thinking>.*?</thinking>", "", task, flags=re.DOTALL)
        # 2. [<thinking>] or [Thinking: ...] anywhere; strip optional preceding em/en-dash
        task = re.sub(r"(\s*[—–]\s*)?\[(?:<|)(?:thinking|Thinking)[^\]]*\]", "", task)
        # 3. Trailing CJK tokens (hallucinated characters like 武功)
        task = re.sub(r"[\u4e00-\u9fff]+\s*$", "", task)

        return task.strip()

    async def execute(
        self,
        task: str,
        role: str = "executor",
        tool_budget: int = 10,
        await_result: bool = True,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the delegation.

        Args:
            task: Task for the delegate to accomplish
            role: Delegate role (researcher, planner, executor, reviewer, tester)
            tool_budget: Maximum tool calls for delegate
            await_result: Whether to wait for completion
            context: Additional context for the delegate
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with delegation outcome
        """
        task = self.sanitize_task(task)
        logger.info(f"Delegating to {role}: {task[:50]}...")

        # Create delegation request
        request = DelegationRequest(
            task=task,
            from_agent=self.parent_agent_id,
            suggested_role=role,
            tool_budget=min(tool_budget, 50),  # Cap at 50
            context=context,
            await_result=await_result,
            parent_goal=self.parent_goal,
        )

        try:
            # Handle the delegation
            response = await self.handler.handle(request)

            if not response.accepted:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Delegation rejected: {response.error}",
                )

            if await_result:
                # Waited for result
                if response.success:
                    result_text = response.result or "Task completed successfully"

                    # Add discoveries if any
                    if response.discoveries:
                        result_text += "\n\n## Key Findings:\n"
                        for discovery in response.discoveries:
                            result_text += f"- {discovery}\n"

                    return ToolResult(
                        success=True,
                        output=result_text,
                        metadata={
                            "delegation_id": response.delegation_id,
                            "delegate_id": response.delegate_id,
                            "tool_calls_used": response.tool_calls_used,
                            "duration_seconds": response.duration_seconds,
                        },
                    )
                else:
                    return ToolResult(
                        success=False,
                        output="",
                        error=response.error or "Delegation failed",
                        metadata={
                            "delegation_id": response.delegation_id,
                            "status": response.status.value,
                        },
                    )
            else:
                # Fire-and-forget
                return ToolResult(
                    success=True,
                    output=f"Delegation started. Delegate ID: {response.delegate_id}",
                    metadata={
                        "delegation_id": response.delegation_id,
                        "delegate_id": response.delegate_id,
                        "async": True,
                    },
                )

        except Exception as e:
            logger.error(f"Delegation failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output="",
                error=f"Delegation error: {str(e)}",
            )


def create_delegate_tool(
    handler: "DelegationHandler",
    parent_agent_id: str = "main",
    parent_goal: Optional[str] = None,
) -> DelegateTool:
    """Factory function to create a DelegateTool.

    Args:
        handler: DelegationHandler instance
        parent_agent_id: ID of the parent agent
        parent_goal: Optional parent goal context

    Returns:
        Configured DelegateTool
    """
    return DelegateTool(
        handler=handler,
        parent_agent_id=parent_agent_id,
        parent_goal=parent_goal,
    )


__all__ = [
    "DelegateTool",
    "create_delegate_tool",
]
