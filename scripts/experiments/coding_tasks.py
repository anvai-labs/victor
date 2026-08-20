# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared multi-turn coding task battery for the A/B harnesses (completion + temperature).

Single-turn QA answers in one turn with no tools, so it can't differentiate completion strategies or
temperatures (every run ties at 1 turn). These tasks instead force the agent to USE TOOLS
(search/grep/read) across several turns against the real Victor repo, then synthesize — exercising the
turn-count / restatement / determinism deltas the A/Bs need:

- a strategy that over-/under-scores completion shows up as extra (restatement) turns;
- a temperature that makes tool-calling flakier shows up as higher turn variance.

All tasks are READ-ONLY (safe to run repeatedly) and answerable from the codebase, with a clear
completion point (the agent should stop once it has read the relevant code).
"""

from __future__ import annotations

from typing import List

# Each requires search/grep + reading 1-3 files, then a concise synthesis (multi-turn, tool-driven).
MULTI_TURN_CODING_TASKS: List[str] = [
    "Find the class that detects repetition/spin in the agentic loop (in victor/agent) and list the "
    "states it can be in. Read the code to confirm, then answer concisely.",
    "Locate the unified temperature resolver under victor/framework/temperature and describe its "
    "source precedence order (highest to lowest). Read the sources file, then summarize.",
    "What values does AgenticLoopConfig.completion_strategy accept? Find the field in "
    "victor/framework/agentic_loop.py and enumerate the options with a one-line meaning each.",
    "Find the function that derives segment-level reward from a trace (victor/framework/rl) and say "
    "which credit-assignment method it uses and why. Read the file to confirm.",
    "Identify the canonical runtime services the AgentOrchestrator delegates to (per CLAUDE.md / the "
    "code). Search the codebase, then list them.",
]
