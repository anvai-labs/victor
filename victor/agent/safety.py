from __future__ import annotations

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

"""Safety module for dangerous operation detection and confirmation.

This module provides:
- Detection of dangerous bash commands
- Detection of destructive file operations
- Risk level categorization
- Confirmation request structures
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Awaitable, Set
import re
import logging

from victor.tools.metadata_registry import get_write_tools as registry_get_write_tools
from victor.tools.tool_names import get_canonical_name

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol
    from victor.core.severity import SeverityLevel

logger = logging.getLogger(__name__)


def _get_risk_icon(
    risk_level: "OperationalRiskLevel",
    presentation: Optional["PresentationProtocol"] = None,
) -> str:
    """Get icon for a risk level using presentation adapter.

    Args:
        risk_level: The operational risk level
        presentation: Optional presentation adapter (creates default if None)

    Returns:
        Icon string for the risk level
    """
    if presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        presentation = create_presentation_adapter()

    # Map risk levels to icon names
    icon_map = {
        OperationalRiskLevel.SAFE: "success",
        OperationalRiskLevel.LOW: "note",
        OperationalRiskLevel.MEDIUM: "warning",
        OperationalRiskLevel.HIGH: "risk_high",
        OperationalRiskLevel.CRITICAL: "risk_critical",
    }

    icon_name = icon_map.get(risk_level, "unknown")
    return presentation.icon(icon_name, with_color=False)


class OperationalRiskLevel(Enum):
    """Risk level categorization for tool/command operations.

    This is the CANONICAL OperationalRiskLevel for operational risk assessment.

    Renamed from RiskLevel to be semantically distinct:
    - OperationalRiskLevel (here): Tool/command risk with SAFE level (5 values)
    - PatternRiskLevel (victor.security.safety.code_patterns): Code pattern risk (4 values)
    """

    SAFE = "safe"  # No risk - read operations
    LOW = "low"  # Minor changes - single file edits
    MEDIUM = "medium"  # Moderate changes - multi-file operations
    HIGH = "high"  # Significant changes - deletions, overwrites
    CRITICAL = "critical"  # System-level destructive operations

    def to_severity(self) -> "SeverityLevel":
        """Convert to unified SeverityLevel."""
        from victor.core.severity import SeverityLevel

        return SeverityLevel.from_operational_risk(self)


class WriteApprovalMode(Enum):
    """Mode for write operation approval.

    Controls when user confirmation is required for file modifications.
    """

    OFF = "off"  # No approval required - auto-approve all operations
    RISKY_ONLY = "risky_only"  # Only require approval for HIGH/CRITICAL risk
    ALL_WRITES = "all_writes"  # Require approval for ALL write operations


# Static fallback for tools that perform write/modify operations.
# PRIMARY source is decorator-driven via @tool(access_mode=AccessMode.WRITE/EXECUTE/MIXED)
# in victor/tools/metadata_registry.py. This fallback ensures backward compatibility
# and covers tools that may not yet have decorator metadata.
# IMPORTANT: Only include actual @tool decorated functions from victor/tools/
_STATIC_WRITE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # Direct file modifications
        "write",  # filesystem.py - writes/overwrites files
        "edit",  # file_editor_tool.py - edits files with transactions
        # Patch/diff application
        "apply_patch",  # patch_tool.py - applies unified diffs
        # Bash execution (can run any command including destructive ones)
        "shell",  # bash.py - executes shell commands
        # Git write operations
        "git",  # git_tool.py - commit, push, reset, etc. (read ops are fine)
        # Refactoring (modifies files in place)
        "refactor_rename_symbol",  # refactor_tool.py
        "refactor_extract_function",  # refactor_tool.py
        "refactor_inline_variable",  # refactor_tool.py
        "refactor_organize_imports",  # refactor_tool.py
        "rename_symbol",  # code_intelligence_tool.py
        # Scaffolding (creates new files/directories)
        "scaffold",  # scaffold_tool.py
        # Batch operations (may include writes)
        "batch",  # batch_processor_tool.py
    }
)


def get_write_tool_names() -> Set[str]:
    """Get all write tool names from registry + static fallback.

    This is the preferred way to get the full set of write tools.
    Returns union of decorator-driven registry and static fallback.

    Returns:
        Set of tool names that perform write/modify operations.
    """
    registry_tools = registry_get_write_tools()
    return registry_tools | set(_STATIC_WRITE_TOOL_NAMES)


# Backward compatibility alias - prefer get_write_tool_names() for dynamic lookup
WRITE_TOOL_NAMES = _STATIC_WRITE_TOOL_NAMES


# Numeric ordering for risk level comparisons
_RISK_ORDER = {
    OperationalRiskLevel.SAFE: 0,
    OperationalRiskLevel.LOW: 1,
    OperationalRiskLevel.MEDIUM: 2,
    OperationalRiskLevel.HIGH: 3,
    OperationalRiskLevel.CRITICAL: 4,
}


@dataclass
class ConfirmationRequest:
    """Request for user confirmation before executing a dangerous operation."""

    tool_name: str
    risk_level: OperationalRiskLevel
    description: str
    details: List[str]
    arguments: Dict[str, Any]

    def format_message(self, presentation: Optional["PresentationProtocol"] = None) -> str:
        """Format confirmation request as a user-friendly message.

        Args:
            presentation: Optional presentation adapter for icons (creates default if None)
        """
        icon = _get_risk_icon(self.risk_level, presentation)

        lines = [
            f"{icon} **{self.risk_level.value.upper()} RISK OPERATION**",
            f"Tool: {self.tool_name}",
            f"Action: {self.description}",
        ]

        if self.details:
            lines.append("Details:")
            for detail in self.details:
                lines.append(f"  - {detail}")

        lines.append("\nProceed with this operation?")
        return "\n".join(lines)


# Confirmation callback type: async function that takes ConfirmationRequest
# and returns True if operation should proceed, False to cancel
ConfirmationCallback = Callable[[ConfirmationRequest], Awaitable[bool]]


class SafetyChecker:
    """Checks operations for dangerous patterns and requests confirmation.

    This class detects potentially dangerous operations and categorizes them
    by risk level. The CLI or UI layer can register a confirmation callback
    to prompt the user before execution.
    """

    # Bash patterns by risk level
    BASH_CRITICAL_PATTERNS = [
        # System destruction
        (r"rm\s+-rf\s+/(?:$|\s)", "Delete entire filesystem"),
        (r"rm\s+-rf\s+/\*", "Delete all root-level directories"),
        (r"dd\s+if=.*\s+of=/dev/sd", "Write directly to disk device"),
        (r"mkfs\.", "Format filesystem"),
        (r":()\{\s*:\|:&\s*\};:", "Fork bomb"),
        (r">\s*/dev/sd[a-z]", "Overwrite disk device"),
        (r"chmod\s+-R\s+777\s+/", "Set world-writable permissions on root"),
    ]

    BASH_HIGH_PATTERNS = [
        # Destructive file operations
        (r"rm\s+-rf\s+", "Recursively delete files/directories"),
        (r"rm\s+-r\s+", "Recursively delete files/directories"),
        (r"rm\s+.*\*", "Delete files with wildcard"),
        (r"rmdir\s+", "Remove directory"),
        # Git destructive operations
        (r"git\s+reset\s+--hard", "Discard all uncommitted changes"),
        (r"git\s+clean\s+-fd", "Delete untracked files"),
        (r"git\s+push\s+.*--force", "Force push (may lose commits)"),
        (r"git\s+push\s+-f", "Force push (may lose commits)"),
        (r"git\s+rebase\s+.*--force", "Force rebase"),
        (r"git\s+branch\s+-D", "Force delete branch"),
        # System changes
        (r"sudo\s+", "Execute with elevated privileges"),
        (r"chmod\s+-R", "Recursively change permissions"),
        (r"chown\s+-R", "Recursively change ownership"),
    ]

    BASH_MEDIUM_PATTERNS = [
        (r"rm\s+", "Delete file(s)"),
        (r"mv\s+.*\s+/dev/null", "Discard file to /dev/null"),
        (r"truncate\s+", "Truncate file"),
        (r">\s+\S+", "Overwrite file with redirection"),
        (r"git\s+checkout\s+--\s+", "Discard changes to file"),
        (r"git\s+stash\s+drop", "Discard stashed changes"),
        (r"pip\s+uninstall", "Uninstall Python package"),
        (r"npm\s+uninstall", "Uninstall npm package"),
    ]

    # File operation patterns
    DANGEROUS_FILE_EXTENSIONS = {
        ".env",
        ".pem",
        ".key",
        ".crt",
        ".p12",
        ".pfx",  # Secrets
        ".db",
        ".sqlite",
        ".sqlite3",  # Databases
        "/etc/passwd",
        "/etc/shadow",  # System files
    }

    def __init__(
        self,
        confirmation_callback: Optional[ConfirmationCallback] = None,
        auto_confirm_low_risk: bool = True,
        require_confirmation_threshold: OperationalRiskLevel = OperationalRiskLevel.HIGH,
        approval_mode: WriteApprovalMode = WriteApprovalMode.RISKY_ONLY,
    ):
        """Initialize safety checker.

        Args:
            confirmation_callback: Async callback for user confirmation
            auto_confirm_low_risk: Auto-approve LOW risk operations
            require_confirmation_threshold: Minimum risk level requiring confirmation
            approval_mode: When to require approval for write operations
                - OFF: Never require approval (dangerous, use for testing only)
                - RISKY_ONLY: Only require for HIGH/CRITICAL risk (default)
                - ALL_WRITES: Require for ALL file modifications (recommended for task mode)
        """
        self.confirmation_callback = confirmation_callback
        self.auto_confirm_low_risk = auto_confirm_low_risk
        self.require_confirmation_threshold = require_confirmation_threshold
        self.approval_mode = approval_mode

        # Compile regex patterns for efficiency
        self._critical_patterns = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in self.BASH_CRITICAL_PATTERNS
        ]
        self._high_patterns = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in self.BASH_HIGH_PATTERNS
        ]
        self._medium_patterns = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in self.BASH_MEDIUM_PATTERNS
        ]

        # Custom patterns from vertical extensions
        self._custom_patterns: List[tuple[re.Pattern, str, OperationalRiskLevel]] = []

    def add_custom_pattern(
        self,
        pattern: str,
        description: str,
        risk_level: str = "HIGH",
        category: str = "custom",
    ) -> None:
        """Add a custom safety pattern from vertical extensions.

        Args:
            pattern: Regex pattern to match.
            description: Human-readable description of the danger.
            risk_level: Risk level string (CRITICAL, HIGH, MEDIUM, LOW).
            category: Category for the pattern (for logging/grouping).
        """
        # Convert risk_level string to OperationalRiskLevel enum
        risk_map = {
            "CRITICAL": OperationalRiskLevel.CRITICAL,
            "HIGH": OperationalRiskLevel.HIGH,
            "MEDIUM": OperationalRiskLevel.MEDIUM,
            "LOW": OperationalRiskLevel.LOW,
            "SAFE": OperationalRiskLevel.SAFE,
        }
        risk = risk_map.get(risk_level.upper(), OperationalRiskLevel.HIGH)

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._custom_patterns.append((compiled, description, risk))
            logger.debug(f"Added custom safety pattern [{category}]: {description}")
        except re.error as e:
            logger.warning(f"Invalid regex pattern for safety checker: {pattern} - {e}")

    def check_bash_command(self, command: str) -> tuple[OperationalRiskLevel, List[str]]:
        """Check a bash command for dangerous patterns.

        Args:
            command: Bash command to check

        Returns:
            Tuple of (risk_level, list of matched pattern descriptions)
        """
        matched: List[str] = []
        max_risk = OperationalRiskLevel.SAFE

        # Check critical patterns first
        for pattern, desc in self._critical_patterns:
            if pattern.search(command):
                matched.append(desc)
                max_risk = OperationalRiskLevel.CRITICAL

        if max_risk == OperationalRiskLevel.CRITICAL:
            return max_risk, matched

        # Check high risk patterns
        for pattern, desc in self._high_patterns:
            if pattern.search(command):
                matched.append(desc)
                if _RISK_ORDER[max_risk] < _RISK_ORDER[OperationalRiskLevel.HIGH]:
                    max_risk = OperationalRiskLevel.HIGH

        if max_risk == OperationalRiskLevel.HIGH:
            return max_risk, matched

        # Check medium risk patterns
        for pattern, desc in self._medium_patterns:
            if pattern.search(command):
                matched.append(desc)
                if _RISK_ORDER[max_risk] < _RISK_ORDER[OperationalRiskLevel.MEDIUM]:
                    max_risk = OperationalRiskLevel.MEDIUM

        # Check custom patterns from vertical extensions
        for pattern, desc, risk in self._custom_patterns:
            if pattern.search(command):
                matched.append(desc)
                if _RISK_ORDER[max_risk] < _RISK_ORDER[risk]:
                    max_risk = risk

        return max_risk, matched

    def check_file_operation(
        self,
        operation: str,
        file_path: str,
        overwrite: bool = False,
    ) -> tuple[OperationalRiskLevel, List[str]]:
        """Check a file operation for risk.

        Args:
            operation: Type of operation (write, edit, delete)
            file_path: Path to the file
            overwrite: Whether operation would overwrite existing content

        Returns:
            Tuple of (risk_level, list of risk descriptions)
        """
        risks: List[str] = []
        max_risk = OperationalRiskLevel.SAFE

        path = Path(file_path)

        # Check for sensitive file types
        for ext in self.DANGEROUS_FILE_EXTENSIONS:
            if file_path.endswith(ext) or ext in file_path:
                risks.append(f"Modifying sensitive file: {ext}")
                max_risk = OperationalRiskLevel.HIGH
                break

        # Check for destructive operations
        if operation == "delete":
            risks.append(f"Deleting file: {file_path}")
            if _RISK_ORDER[max_risk] < _RISK_ORDER[OperationalRiskLevel.HIGH]:
                max_risk = OperationalRiskLevel.HIGH
        elif operation == "write" and overwrite:
            # Check if file exists (would be overwritten)
            if path.exists():
                risks.append(f"Overwriting existing file: {file_path}")
                if _RISK_ORDER[max_risk] < _RISK_ORDER[OperationalRiskLevel.MEDIUM]:
                    max_risk = OperationalRiskLevel.MEDIUM

        # Check for system paths
        if file_path.startswith("/etc/") or file_path.startswith("/usr/"):
            risks.append("Modifying system file")
            max_risk = OperationalRiskLevel.HIGH

        return max_risk, risks

    def is_write_tool(self, tool_name: str) -> bool:
        """Check if a tool is a write/modify operation.

        Uses decorator-driven registry as primary source, with static fallback
        for backward compatibility during migration.

        Args:
            tool_name: Name of the tool

        Returns:
            True if the tool can modify files/state
        """
        # Primary: check decorator-driven registry
        canonical_tool_name = get_canonical_name(tool_name)
        registry_write_tools = {get_canonical_name(name) for name in registry_get_write_tools()}
        if canonical_tool_name in registry_write_tools:
            return True
        # Fallback: check static list for tools without decorator metadata
        return canonical_tool_name in _STATIC_WRITE_TOOL_NAMES

    async def check_and_confirm(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Check operation safety and request confirmation if needed.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments

        Returns:
            Tuple of (should_proceed, optional_rejection_reason)
        """
        canonical_tool_name = get_canonical_name(tool_name)

        # Fast path: approval mode OFF - auto-approve everything
        if self.approval_mode == WriteApprovalMode.OFF:
            return True, None

        risk_level = OperationalRiskLevel.SAFE
        descriptions: List[str] = []
        details: List[str] = []
        is_write_operation = self.is_write_tool(canonical_tool_name)

        # Check bash commands
        if canonical_tool_name == "shell":
            command = arguments.get("cmd") or arguments.get("command", "")
            risk_level, details = self.check_bash_command(command)
            if details:
                descriptions.append(f"Execute: {command[:100]}...")

        # Check file write operations
        elif canonical_tool_name == "write":
            file_path = arguments.get("path", arguments.get("file_path", ""))
            risk_level, details = self.check_file_operation("write", file_path, overwrite=True)
            if details:
                descriptions.append(f"Write to: {file_path}")

        # Check file edit operations
        elif canonical_tool_name == "edit":
            edits = arguments.get("ops") or arguments.get("edits") or arguments.get("files") or []
            for edit in edits:
                if not isinstance(edit, dict):
                    continue
                file_path = edit.get("path", "")
                edit_risk, edit_details = self.check_file_operation("edit", file_path)
                if _RISK_ORDER[edit_risk] > _RISK_ORDER[risk_level]:
                    risk_level = edit_risk
                details.extend(edit_details)
            if edits:
                descriptions.append(f"Edit {len(edits)} file(s)")

        # Check git operations
        elif tool_name == "git":
            subcommand = arguments.get("subcommand", "")
            args = arguments.get("args", "")
            full_cmd = f"git {subcommand} {args}"
            risk_level, details = self.check_bash_command(full_cmd)
            if details:
                descriptions.append(f"Git: {subcommand}")

        # Determine if confirmation is needed
        requires_confirmation = False

        # ALL_WRITES mode: require confirmation for any write operation
        if self.approval_mode == WriteApprovalMode.ALL_WRITES and is_write_operation:
            requires_confirmation = True
            # If no descriptions yet, add a generic one
            if not descriptions:
                descriptions.append(f"Execute write operation: {tool_name}")
            # Set minimum risk to LOW for display purposes
            if risk_level == OperationalRiskLevel.SAFE:
                risk_level = OperationalRiskLevel.LOW
                details.append("Write operation requiring approval")

        # RISKY_ONLY mode: only require confirmation for high-risk operations
        elif self.approval_mode == WriteApprovalMode.RISKY_ONLY:
            if _RISK_ORDER[risk_level] >= _RISK_ORDER[self.require_confirmation_threshold]:
                requires_confirmation = True

        # Auto-approve if confirmation not required
        if not requires_confirmation:
            return True, None

        # No callback registered - log warning but proceed
        if not self.confirmation_callback:
            if risk_level in (OperationalRiskLevel.HIGH, OperationalRiskLevel.CRITICAL):
                logger.warning(
                    f"High-risk operation without confirmation callback: "
                    f"{tool_name} - {', '.join(details)}"
                )
            elif self.approval_mode == WriteApprovalMode.ALL_WRITES:
                logger.warning(
                    f"Write operation without confirmation callback in ALL_WRITES mode: "
                    f"{tool_name}"
                )
            return True, None

        # Create confirmation request
        request = ConfirmationRequest(
            tool_name=canonical_tool_name,
            risk_level=risk_level,
            description=(
                "; ".join(descriptions) if descriptions else f"Execute {canonical_tool_name}"
            ),
            details=details,
            arguments=arguments,
        )

        # Request confirmation
        try:
            confirmed = await self.confirmation_callback(request)
            if confirmed:
                return True, None
            else:
                return False, f"Operation cancelled by user: {request.description}"
        except Exception as e:
            logger.error(f"Confirmation callback failed: {e}")
            # If callback fails, block high-risk operations
            if risk_level in (OperationalRiskLevel.HIGH, OperationalRiskLevel.CRITICAL):
                return False, f"Confirmation failed: {e}"
            return True, None


# Default singleton instance
_default_checker: Optional[SafetyChecker] = None


def _resolve_approval_mode(mode_str: str) -> WriteApprovalMode:
    """Resolve approval mode from string setting."""
    mode_map = {
        "off": WriteApprovalMode.OFF,
        "risky_only": WriteApprovalMode.RISKY_ONLY,
        "all_writes": WriteApprovalMode.ALL_WRITES,
    }
    return mode_map.get(mode_str.lower(), WriteApprovalMode.RISKY_ONLY)


def get_safety_checker() -> SafetyChecker:
    """Get the default safety checker instance.

    The approval mode is configured via settings.write_approval_mode:
    - "off": No approval required (dangerous)
    - "risky_only": Only HIGH/CRITICAL risk (default)
    - "all_writes": All write operations require approval
    """
    global _default_checker
    if _default_checker is None:
        # Try to get approval mode from settings
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            approval_mode = _resolve_approval_mode(settings.security.write_approval_mode)
        except Exception:
            # Default to RISKY_ONLY if settings unavailable
            approval_mode = WriteApprovalMode.RISKY_ONLY

        _default_checker = SafetyChecker(approval_mode=approval_mode)
    return _default_checker


def reset_safety_checker() -> None:
    """Reset the global safety-checker singleton so it is rebuilt on next use.

    The singleton caches its approval mode (from settings at first creation) and
    accumulates mutable state — a confirmation callback (``set_confirmation_callback``)
    and custom deny patterns (added via the capability registry). Nothing resets it
    between test runs, so a test that leaves it in a denying/mutated state silently
    blocks unrelated tools in later tests (an order-dependent failure). The test
    suite calls this via an autouse fixture; production never needs it.
    """
    global _default_checker
    _default_checker = None


def set_confirmation_callback(callback: ConfirmationCallback) -> None:
    """Set the global confirmation callback for dangerous operations.

    Args:
        callback: Async function that takes ConfirmationRequest and returns bool
    """
    checker = get_safety_checker()
    checker.confirmation_callback = callback


# =============================================================================
# HITL Integration
# =============================================================================
# These functions bridge SafetyChecker with the Human-in-the-Loop workflow system,
# enabling consistent confirmation UX across CLI/API.


def create_hitl_confirmation_callback(
    hitl_handler: Optional[Any] = None,
    timeout: Optional[float] = None,
    fallback: Optional[str] = None,
) -> ConfirmationCallback:
    """Create a HITL-backed confirmation callback for SafetyChecker.

    This bridges the SafetyChecker's confirmation mechanism with the HITL workflow
    system, enabling consistent UI handling across CLI and API modes.

    Args:
        hitl_handler: HITLHandler implementation (defaults to DefaultHITLHandler)
        timeout: Override timeout in seconds (default from settings)
        fallback: Override fallback behavior ("abort", "continue", "skip")

    Returns:
        Async callback suitable for SafetyChecker.confirmation_callback
    """
    from victor.workflows.hitl import (
        HITLRequest,
        HITLNodeType,
        HITLFallback,
        HITLStatus,
        DefaultHITLHandler,
    )
    import asyncio
    import uuid

    # Resolve handler
    if hitl_handler is None:
        hitl_handler = DefaultHITLHandler()

    # Resolve settings defaults
    default_timeout = 300.0
    default_fallback = "abort"

    try:
        from victor.config.settings import load_settings

        settings = load_settings()
        default_timeout = getattr(settings, "hitl_default_timeout", 300.0)
        default_fallback = getattr(settings, "hitl_default_fallback", "abort")
    except Exception as e:
        logger.debug("Failed to load HITL settings, using defaults: %s", e)

    effective_timeout = timeout if timeout is not None else default_timeout
    effective_fallback = fallback if fallback is not None else default_fallback

    # Map fallback string to enum
    fallback_map = {
        "abort": HITLFallback.ABORT,
        "continue": HITLFallback.CONTINUE,
        "skip": HITLFallback.SKIP,
        "retry": HITLFallback.RETRY,
    }
    fallback_enum = fallback_map.get(effective_fallback.lower(), HITLFallback.ABORT)

    async def hitl_confirmation_callback(request: ConfirmationRequest) -> bool:
        """Convert SafetyChecker request to HITL and get response."""
        # Get risk icon using presentation adapter
        risk_icon = _get_risk_icon(request.risk_level)

        # Create HITL request
        hitl_request = HITLRequest(
            request_id=f"safety_{uuid.uuid4().hex[:12]}",
            node_id="safety_check",
            hitl_type=HITLNodeType.APPROVAL,
            prompt=f"{risk_icon} {request.risk_level.value.upper()} RISK: {request.description}",
            context={
                "tool": request.tool_name,
                "risk_level": request.risk_level.value,
                "details": request.details,
                "arguments": {k: str(v)[:200] for k, v in request.arguments.items()},
            },
            timeout=effective_timeout,
            fallback=fallback_enum,
        )

        try:
            # Request human input with timeout
            response = await asyncio.wait_for(
                hitl_handler.request_human_input(hitl_request),
                timeout=effective_timeout,
            )

            # Return approval status
            if response.status == HITLStatus.APPROVED:
                return True
            elif response.status == HITLStatus.TIMEOUT:
                # On timeout, use fallback behavior
                if fallback_enum == HITLFallback.CONTINUE:
                    logger.warning(
                        f"Safety confirmation timed out, continuing: {request.tool_name}"
                    )
                    return True
                else:
                    logger.warning(f"Safety confirmation timed out, aborting: {request.tool_name}")
                    return False
            else:
                # Rejected or other status
                logger.info(
                    f"Operation rejected: {request.tool_name} - {response.reason or 'no reason'}"
                )
                return False

        except asyncio.TimeoutError:
            logger.warning(f"HITL timeout for safety confirmation: {request.tool_name}")
            if fallback_enum == HITLFallback.CONTINUE:
                return True
            return False

        except Exception as e:
            logger.error(f"HITL confirmation error: {e}")
            # Block high-risk operations on error
            if request.risk_level in (
                OperationalRiskLevel.HIGH,
                OperationalRiskLevel.CRITICAL,
            ):
                return False
            return True

    return hitl_confirmation_callback


def setup_hitl_safety_integration(hitl_handler: Optional[Any] = None) -> None:
    """Configure SafetyChecker to use HITL for confirmations.

    Call this once during application startup to enable HITL-backed
    confirmation prompts for dangerous operations.

    Args:
        hitl_handler: Optional custom HITLHandler implementation
    """
    callback = create_hitl_confirmation_callback(hitl_handler)
    set_confirmation_callback(callback)
    logger.info("SafetyChecker configured with HITL confirmation handler")
