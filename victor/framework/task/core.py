"""Task - What the agent should accomplish.

This module defines Task and TaskResult classes for representing
agent work and its outcomes.

Example:
    # Simple usage - just pass a string to agent.run()
    result = await agent.run("Write a hello world function")

    # Advanced usage - use Task for more control
    task = Task(
        prompt="Refactor the auth module",
        type=FrameworkTaskType.EDIT,
        files=["src/auth.py"],
        constraints={"no_delete": True}
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from victor.tools.core_tool_aliases import canonicalize_core_tool_name


class FrameworkTaskType(Enum):
    """Framework-level task types for agent operation.

    High-level types used by the framework Task class.
    Uses auto() values for internal identity.

    Renamed from TaskType to be semantically distinct:
    - TaskType (victor.classification.pattern_registry): Canonical prompt classification
    - TrackerTaskType: Progress tracking with milestones
    - LoopDetectorTaskType: Loop detection thresholds
    - ClassifierTaskType: Unified classification output
    - FrameworkTaskType: Framework-level task abstraction
    """

    CHAT = auto()
    """General conversation or Q&A."""

    ANALYZE = auto()
    """Code analysis, review, or explanation."""

    EDIT = auto()
    """Modify existing files."""

    CREATE = auto()
    """Create new files or content."""

    SEARCH = auto()
    """Search through code or files."""

    EXECUTE = auto()
    """Run commands or scripts."""

    WORKFLOW = auto()
    """Multi-step coordinated workflow."""


@dataclass
class Task:
    """Definition of a task for the agent.

    Tasks encapsulate what the agent should do, with optional
    type hints, file scope, and constraints.

    For simple use cases, just pass a string to agent.run().
    Use Task when you need more control over execution.

    Attributes:
        prompt: What the agent should do (required)
        type: Type of task for optimization hints
        files: Relevant files to focus on
        context: Additional context dictionary
        constraints: Execution constraints
        tool_budget: Override default tool budget
        max_iterations: Override max iterations

    Example:
        # Simple - most users just need this
        result = await agent.run("Fix the bug")

        # Advanced - explicit Task for more control
        task = Task(
            prompt="Refactor the auth module",
            type=FrameworkTaskType.EDIT,
            files=["src/auth.py", "src/auth_utils.py"],
            constraints={"preserve_api": True}
        )
    """

    prompt: str
    type: FrameworkTaskType = FrameworkTaskType.CHAT
    files: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Advanced options
    tool_budget: Optional[int] = None
    max_iterations: Optional[int] = None


@dataclass
class TaskResult:
    """Result from executing a task.

    Contains the agent's response, tool calls made,
    and metadata about the execution.

    Attributes:
        content: The agent's text response
        tool_calls: List of tool calls made during execution
        success: Whether the task completed successfully
        error: Error message if task failed
        metadata: Additional execution metadata
        status: Outcome status — "ok" (completed) or "awaiting_approval" (durably
            paused on a policy ASK, FEP-0029). Defaults to "ok".
        run_id: Resume token, present only when ``status == "awaiting_approval"``.
        approval_request: The pending ApprovalRequest (as a dict), present only when
            ``status == "awaiting_approval"``.

    Example:
        result = await agent.run("Create a new module")

        print(result.content)
        print(f"Modified: {result.files_modified}")
        print(f"Tools used: {len(result.tool_calls)}")
    """

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # FEP-0029: durable pause outcome. "ok" for a normal turn; "awaiting_approval" when the turn
    # parked on a policy ASK, with run_id + approval_request set for resume.
    status: str = "ok"
    run_id: Optional[str] = None
    approval_request: Optional[Dict[str, Any]] = None

    @property
    def files_modified(self) -> List[str]:
        """Get list of files modified during task execution.

        Returns:
            List of file paths that were written or edited
        """
        modified = []
        for call in self.tool_calls:
            tool = canonicalize_core_tool_name(call.get("tool", call.get("tool_name", "")))
            if tool in ("write", "edit", "file_edit") and call.get("success", True):
                path = call.get("arguments", {}).get("path") or call.get("arguments", {}).get(
                    "file_path"
                )
                if path and path not in modified:
                    modified.append(path)
        return modified

    @property
    def files_read(self) -> List[str]:
        """Get list of files read during task execution.

        Returns:
            List of file paths that were read
        """
        read = []
        for call in self.tool_calls:
            tool = canonicalize_core_tool_name(call.get("tool", call.get("tool_name", "")))
            if tool in ("read", "ls") and call.get("success", True):
                path = call.get("arguments", {}).get("path") or call.get("arguments", {}).get(
                    "file_path"
                )
                if path and path not in read:
                    read.append(path)
        return read

    @property
    def commands_executed(self) -> List[str]:
        """Get list of shell commands executed during task.

        Returns:
            List of command strings that were run
        """
        commands = []
        for call in self.tool_calls:
            tool = call.get("tool", call.get("tool_name", ""))
            if tool in ("shell", "bash", "run_command"):
                cmd = call.get("arguments", {}).get("command")
                if cmd:
                    commands.append(cmd)
        return commands

    @property
    def tool_count(self) -> int:
        """Get total number of tool calls made.

        Returns:
            Count of tool calls
        """
        return len(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "files_modified": self.files_modified,
            "files_read": self.files_read,
            "tool_count": self.tool_count,
        }
