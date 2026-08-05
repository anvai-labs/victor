# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework-level verification protocol (FEP-0018).

Defines the ``Verifier`` protocol — a pluggable component that verifies the
agent's work after it claims done and provides structured feedback for retry.
Any agent session (interactive chat, benchmark eval, workflow, CLI) can use it.

Zero victor-package dependencies (stdlib only) so any module can import it
without circular-import risk.

Usage:
    class MyVerifier:
        async def verify(self, *, workspace=None, state=None) -> VerificationResult:
            ...

    loop = AgenticLoop(..., verifier=MyVerifier(), max_verify_retries=2)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying the agent's work."""

    passed: int
    """Number of checks/tests that passed."""

    total: int
    """Total number of checks/tests run."""

    raw_output: str = ""
    """Full verifier output (test logs, lint results, etc.)."""

    feedback: str = ""
    """Human-readable summary injected into the agent's retry turn
    (e.g., "Verification: 16/17 tests passed. FAILED: test_foo")."""

    @property
    def is_verified(self) -> bool:
        """True when all checks passed."""
        return self.total > 0 and self.passed == self.total


@runtime_checkable
class Verifier(Protocol):
    """Verify the agent's work after it claims done.

    The agentic loop calls ``verify()`` after DECIDE=COMPLETE (and before
    exiting). If the result is not verified, the loop injects the feedback
    and re-enters (CONTINUE), bounded by ``max_verify_retries``.
    """

    async def verify(
        self,
        *,
        workspace: Optional[Path] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Run the verification.

        Args:
            workspace: The agent's working directory (for git diff / file checks).
            state: The loop's state dict (task_type, perception, etc.).

        Returns:
            A ``VerificationResult`` with passed/total counts + feedback.
        """
        ...
