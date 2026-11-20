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

from typing import List

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
    context: dict = None,
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
        context: Tool context provided by orchestrator

    Returns:
        Operation result string
    """
    if context is None:
        return "Error: Context not provided."

    sandbox_instance: CodeSandbox = context.get("code_manager")
    if not sandbox_instance:
        return "Error: CodeSandbox not found in context."

    op = operation.lower().strip()

    if op == "execute":
        if not code:
            return "Error: 'code' parameter required for execute operation."
        return await _execute_code(sandbox_instance, code)

    elif op == "upload":
        if not file_paths:
            return "Error: 'file_paths' parameter required for upload operation."
        return await _upload_files(sandbox_instance, file_paths)

    else:
        return f"Error: Unknown operation '{operation}'. Use 'execute' or 'upload'."
