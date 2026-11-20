from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Docker sandbox tool for executing Python code (the ``sandbox`` tool).

FOU-S2 (ADR-035 item 5): the ``CodeSandbox`` lifecycle quality bar — lazy
acquire, owned/finalized, self-expiring idle TTL, network disabled by
default, resource caps, owner-labelled and reapable — is owned by core
(:mod:`victor.tools.sandbox.container_sandbox`); this module is the vertical
tool surface on top of it. ``CodeSandbox`` below *is* the core
:class:`~victor.tools.sandbox.container_sandbox.ContainerSandbox` class
(identity, not a copy), so both surfaces share one lifecycle, one reaper and
one owner-label vocabulary. The historical module-level names are re-exported
for backward compatibility (session bootstrap leak guard, cleanup-containers
command, evaluation reaper).
"""

from typing import Any, Dict, List

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.sandbox.container_sandbox import (
    DEFAULT_IDLE_TTL_SECONDS as DEFAULT_IDLE_TTL_SECONDS,
    DEFAULT_IMAGE as DEFAULT_IMAGE,
    DOCKER_AVAILABLE as DOCKER_AVAILABLE,
    IDLE_TTL_LABEL as IDLE_TTL_LABEL,
    LEGACY_AUTO_REAP_MIN_AGE_SECONDS as LEGACY_AUTO_REAP_MIN_AGE_SECONDS,
    LEASE_PATH as LEASE_PATH,
    OWNER_HOST_LABEL as OWNER_HOST_LABEL,
    OWNER_PID_LABEL as OWNER_PID_LABEL,
    OWNER_START_LABEL as OWNER_START_LABEL,
    SANDBOX_CONTAINER_LABEL as SANDBOX_CONTAINER_LABEL,
    SANDBOX_CONTAINER_VALUE as SANDBOX_CONTAINER_VALUE,
    ContainerSandbox,
    ContainerSandboxUnavailableError,
    created_age_seconds as created_age_seconds,
    execute_code,
    find_orphaned_sandboxes as find_orphaned_sandboxes,
    host_identity as host_identity,
    owner_alive as owner_alive,
    owner_gone as owner_gone,
    owner_labels as owner_labels,
    process_start_marker as process_start_marker,
    reap_orphaned_sandboxes as reap_orphaned_sandboxes,
    upload_files,
    watchdog_command as watchdog_command,
)

# Compat surface: names this module historically owned. ``CodeSandbox`` and
# ``ResourceError`` are identity aliases of the core types — not subclasses —
# so isinstance checks and lifecycle behavior are identical on both surfaces.
CodeSandbox = ContainerSandbox
ResourceError = ContainerSandboxUnavailableError

# Async helpers, shared formatting contract with core.
_execute_code = execute_code
_upload_files = upload_files


def _sandbox_failure(message: str, recovery_hint: str) -> Dict[str, Any]:
    """Build the typed failure dict the @tool adapter converts to a failed
    ``ToolResult``.

    Codex review (code executor context contract): the tool used to return
    "Error: ..." STRINGS on missing context, which the function-tool adapter
    wrapped as ``ToolResult(success=True)`` — a success-shaped report with
    zero sandbox work done. Every refusal below is now a structured failure
    dict (``success=False`` + ``category`` + ``recovery_hint``, the same
    vertical convention as ``graph_tool``) so the adapter publishes
    ``success=False`` with typed metadata.
    """
    return {
        "success": False,
        "error": message,
        "category": "tool_execution",
        "recovery_hint": recovery_hint,
    }


@tool(
    category="execution",
    priority=Priority.MEDIUM,  # Task-specific code execution
    access_mode=AccessMode.EXECUTE,  # Runs code in container
    danger_level=DangerLevel.HIGH,  # Arbitrary code execution
    keywords=["sandbox", "execute", "python", "container", "docker", "upload", "code"],
)
async def sandbox(
    operation: str,
    code: str = "",
    file_paths: List[str] = None,
    _exec_ctx: Dict[str, Any] = None,
) -> str:
    """Unified sandbox operations for code execution in isolated Docker container.

    The sandbox container is created on first use and removed after a period
    of inactivity; files uploaded before it expires are not kept.

    Operations:
    - "execute": Run Python code in the sandbox
    - "upload": Upload local files to the sandbox

    Args:
        operation: Operation to perform - "execute" or "upload"
        code: Python code to execute (for "execute" operation)
        file_paths: List of local file paths to upload (for "upload" operation)
        _exec_ctx: Reserved framework execution context injected by the tool
            executor (carries the session's ``code_manager``). Never supplied
            by the model.

    Returns:
        Operation result string, or a structured failure dict
        (``success=False`` + ``category`` + ``recovery_hint``) when the
        execution context or session sandbox handle is missing or an
        operation argument is invalid — never a success-shaped error
        string. The function-tool adapter converts the dict to a failed
        ``ToolResult``.
    """
    context = _exec_ctx
    if not isinstance(context, dict):
        return _sandbox_failure(
            "sandbox requires the runtime execution context (injected by the "
            "tool executor as _exec_ctx), which was missing on this dispatch.",
            recovery_hint=(
                "Dispatch the sandbox tool through the normal tool-execution "
                "path; do not pass internal context arguments yourself."
            ),
        )

    sandbox_instance: CodeSandbox = context.get("code_manager")
    if not sandbox_instance:
        return _sandbox_failure(
            "sandbox is unavailable: no code-execution sandbox handle "
            "('code_manager') in this session's execution context.",
            recovery_hint=(
                "The container runtime is unavailable or disabled in this "
                "session; run this code with the shell tool or enable Docker."
            ),
        )

    op = operation.lower().strip()

    if op == "execute":
        if not code:
            return _sandbox_failure(
                "'code' parameter required for execute operation.",
                recovery_hint="Pass the Python source to run in the 'code' argument.",
            )
        return await _execute_code(sandbox_instance, code)

    elif op == "upload":
        if not file_paths:
            return _sandbox_failure(
                "'file_paths' parameter required for upload operation.",
                recovery_hint="Pass the local paths to copy in 'file_paths'.",
            )
        return await _upload_files(sandbox_instance, file_paths)

    else:
        return _sandbox_failure(
            f"Unknown operation '{operation}'. Use 'execute' or 'upload'.",
            recovery_hint="Retry with operation='execute' or operation='upload'.",
        )
