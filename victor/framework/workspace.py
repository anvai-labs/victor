# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework-level workspace utilities.

Provides ground-truth operations on a workspace's git state — capturing what
the agent changed, resetting to a known commit, etc. These are generic
capabilities useful across the framework (agentic loop, fulfillment detection,
evaluation, middleware, tools), not just the benchmark adapter.

Has ZERO victor-package dependencies (stdlib only) so any module can import it
without pulling in the orchestrator/tool/container dependency chain.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def workspace_git_diff(workspace: Path, timeout: int = 30) -> str:
    """Capture the agent's file changes as a unified diff from the workspace's
    git state — the ground truth.

    Stages all changes (``git add -A``) then diffs staged vs HEAD
    (``git diff --cached HEAD``), capturing modified, new, AND deleted files.
    This is more reliable than hook-based edit-capture (which misses shell-based
    edits, stale snapshots, and some new files).

    Use cases:
    - Evaluation: the canonical patch source for final scoring + verify-and-retry.
    - Agentic loop: progress tracking ("what has the agent changed so far?").
    - Fulfillment detection: grounded "did the agent modify the relevant files?".
    - Middleware/observability: per-turn diff for debugging.

    Args:
        workspace: Path to the git workspace (the agent's working directory).
        timeout: Max seconds for each git subprocess.

    Returns:
        Unified diff string, or ``""`` if the workspace is not a git repo /
        git is unavailable.
    """
    try:
        add_proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(workspace),
            "add",
            "-A",
            "--",
            ".",
            ":!.victor",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(add_proc.communicate(), timeout=timeout)

        diff_proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(workspace),
            "diff",
            "--cached",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _err = await asyncio.wait_for(diff_proc.communicate(), timeout=timeout)
        return out.decode("utf-8", "replace")
    except Exception:
        return ""


async def workspace_files_modified(workspace: Path, timeout: int = 15) -> list[str]:
    """Return the list of files changed vs HEAD (modified + new + deleted).

    A lightweight companion to :func:`workspace_git_diff` for metrics and
    progress signals (e.g., "the agent modified 3 files this turn").
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(workspace),
            "diff",
            "--name-only",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return [
            line.strip() for line in out.decode("utf-8", "replace").splitlines() if line.strip()
        ]
    except Exception:
        return []
