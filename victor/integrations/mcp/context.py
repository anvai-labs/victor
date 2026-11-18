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

"""Sandboxed MCP execution context (FEP-0049-followup B2).

Historically, MCP tool calls ran with NO sandbox admission: the MCP server
passed an empty execution context, so tools fell back to a fresh settings
read at spawn time (the historical fail-open posture). This module models
the launch-time admission on ``_admit_git_policy`` (api/routes/git_routes.py):
contextvar → ``RunPolicySnapshot.from_settings(get_settings())`` → typed
failure. The admitted snapshot is injected into every ``tools/call`` so the
downstream enforcement points (``bash.py`` ``_admitted_policy``,
``subprocess_executor``'s ``resolve_required_launch``, ...) consume the same
policy the launch admitted — never a re-read of mutable settings.

Admission NEVER degrades to an open posture on error: a settings read or
snapshot failure raises :class:`MCPAdmissionError` and the entry points
refuse tool execution (fail-closed) while the server itself still starts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from victor.tools.sandbox.backends import RequiredLaunchDecision

SANDBOX_ESCAPE_HATCH = (
    "Escape hatch: set sandbox_policy: 'disabled' in settings or set "
    "VICTOR_SANDBOX_POLICY=disabled in the environment."
)


class MCPAdmissionError(RuntimeError):
    """MCP launch admission could not establish an enforceable posture.

    Callers must refuse tool execution (isError results) without running
    anything; this is the fail-closed boundary, distinct from ordinary tool
    failures. The message carries the actionable escape hatch.
    """


@dataclass(frozen=True)
class MCPToolContext:
    """Immutable admission result injected into every MCP tool call.

    Attributes:
        run_policy_snapshot: The admitted immutable policy snapshot.
        workspace_root: Working directory the server was launched in.
        backend_decision: Canonical required-launch decision (refuse marks a
            required policy that cannot be enforced — callers must refuse).
        settings: The settings object the admission read, so tools that read
            settings from the execution context observe the SAME read the
            admission used (consistency, no divergent re-read).
    """

    # Duck-typed runtime RunPolicySnapshot: the only contract is
    # ``to_sandbox_settings()``. Imported nowhere in this module (layering).
    run_policy_snapshot: Any
    workspace_root: Path
    backend_decision: RequiredLaunchDecision
    settings: Any = None

    def to_exec_context(self) -> dict[str, Any]:
        """Build the ``_exec_ctx`` dict downstream tools already consume."""
        return {
            "run_policy_snapshot": self.run_policy_snapshot,
            "workspace_root": str(self.workspace_root),
            "settings": self.settings,
        }

    def refusal_text(self) -> str:
        """The actionable refusal for an unenforceable required policy."""
        reason = self.backend_decision.reason or ("sandbox policy is required but unenforceable")
        if SANDBOX_ESCAPE_HATCH not in reason:
            reason = f"{reason} {SANDBOX_ESCAPE_HATCH}"
        return f"MCP tool execution refused: {reason}"


def admit_mcp_execution_context(
    workspace_root: Path,
    snapshot: Any,
    *,
    resolve_required_launch: Any | None = None,
) -> MCPToolContext:
    """Admit the execution context for an MCP server launch.

    Modeled on ``_admit_git_policy``. Dependency-inverted per the layering
    ratchet: the CALLER (client layer) builds the runtime policy snapshot
    (contextvar -> ``RunPolicySnapshot.from_settings``) and injects it here;
    this module never imports ``victor.runtime``. Only the sandbox launch
    decision is resolved here (same infrastructure layer).

    Args:
        workspace_root: Working directory the server was launched in.
        snapshot: The runtime policy snapshot (duck-typed via
            ``to_sandbox_settings()``).

    Raises:
        Nothing from admission itself. Callers translate snapshot-build
        failures into an ``admission_refusal`` string (fail-closed: no
        context, every call refused).
    """
    if resolve_required_launch is None:
        # Late import as the declared fallback: this module sits below the
        # sandbox layer, and tests inject the resolver explicitly.
        from victor.tools.sandbox import resolve_required_launch as _resolve

        resolve_required_launch = _resolve

    backend_decision = resolve_required_launch(snapshot.to_sandbox_settings())
    return MCPToolContext(
        run_policy_snapshot=snapshot,
        workspace_root=workspace_root,
        backend_decision=backend_decision,
    )
