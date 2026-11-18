"""Sandbox/isolation configuration for safe tool execution."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SandboxSettings(BaseModel):
    """Sandbox/isolation configuration for safe tool execution.

    Controls filesystem isolation, process namespace restrictions,
    and environment variable filtering when executing tools.

    ``sandbox_policy`` selects the containment posture (ADR-035):

    * ``None`` (default) — derive the **fail-closed ``required``** posture
      (the ``VICTOR_SANDBOX_POLICY`` environment variable may override).
      The legacy ``sandbox_enabled`` bool no longer selects a posture.
    * ``disabled`` — explicit opt-out: no isolation, with a one-time
      unsandboxed warning.
    * ``preferred`` — explicit opt-in convenience mode: isolate when a
      backend is available; if it is missing or the platform is
      unsupported, run unsandboxed and record the degradation.
    * ``required`` — isolation is mandatory: when no enforcing backend is
      available, tool dispatch must not execute (fail closed).
    """

    sandbox_enabled: bool = False
    sandbox_policy: Optional[Literal["disabled", "preferred", "required"]] = None
    sandbox_filesystem_mode: str = "workspace-only"
    sandbox_namespace_restrictions: bool = True
    sandbox_network_isolation: bool = True
    sandbox_seccomp: bool = True
    sandbox_allowed_mounts: List[str] = Field(default_factory=list)
