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

"""Sandboxing for Victor tools.

Two isolation models live here:

* :mod:`victor.tools.sandbox.backends` — OS-level launchers that wrap a host
  subprocess argv (bwrap/Seatbelt/none). Activated via ``settings.sandbox``
  (``sandbox_policy``: ``disabled`` / ``preferred`` / ``required``);
  ``preferred`` resolution is fail-open, ``required`` is fail-closed.
* :mod:`victor.tools.sandbox.container_sandbox` — the Docker container
  sandbox for code execution (FOU-S2, ADR-035 item 5): the promoted
  ``CodeSandbox`` lifecycle quality bar, gated behind
  :func:`resolve_container_sandbox` (None when Docker is unavailable, typed
  refusal under ``required``).
"""

from victor.tools.sandbox.backends import (
    BubblewrapSandbox,
    NoneSandbox,
    RequiredLaunchDecision,
    SandboxBackend,
    PolicyUnavailable,
    SandboxRequiredUnavailableError,
    SeatbeltSandbox,
    _effective_policy_from_settings,
    build_seccomp_deny_program,
    current_sandbox_posture,
    effective_sandbox_policy,
    force_unsandboxed_for_session,
    resolve_required_launch,
    resolve_sandbox_backend,
)
from victor.tools.sandbox.container_sandbox import (
    ContainerSandbox,
    ContainerSandboxUnavailableError,
    daemon_available,
    find_orphaned_sandboxes,
    reap_orphaned_sandboxes,
    resolve_container_sandbox,
    sandbox_kwargs_from_settings,
    watchdog_command,
)
from victor.tools.sandbox.launch import (
    SandboxLaunch,
    close_pass_fds,
    prepare_launch,
    run_launch,
    run_launch_async,
)

__all__ = [
    "SandboxBackend",
    "NoneSandbox",
    "BubblewrapSandbox",
    "SeatbeltSandbox",
    "PolicyUnavailable",
    "SandboxRequiredUnavailableError",
    "RequiredLaunchDecision",
    "_effective_policy_from_settings",
    "SandboxLaunch",
    "close_pass_fds",
    "prepare_launch",
    "run_launch",
    "run_launch_async",
    "build_seccomp_deny_program",
    "current_sandbox_posture",
    "effective_sandbox_policy",
    "force_unsandboxed_for_session",
    "resolve_required_launch",
    "resolve_sandbox_backend",
    "ContainerSandbox",
    "ContainerSandboxUnavailableError",
    "daemon_available",
    "find_orphaned_sandboxes",
    "reap_orphaned_sandboxes",
    "resolve_container_sandbox",
    "sandbox_kwargs_from_settings",
    "watchdog_command",
]
