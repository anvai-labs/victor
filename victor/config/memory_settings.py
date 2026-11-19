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

"""Core long-term memory settings (FEP-0048)."""

from __future__ import annotations

from pydantic import BaseModel


class MemorySettings(BaseModel):
    """Core long-term memory behavior (FEP-0048).

    Write-path defaults are conservative: wrong memories actively harm every
    later session, so capture is human-confirmed unless explicitly opted in.

    Read-path (PERCEIVE injection, FOU-A2c) defaults are ON with off-when-empty
    semantics: an empty store produces zero prompt change and zero token cost,
    so enabling injection is behaviorally invisible until the first memory is
    captured. Whether the default-on flip STAYS on is decided by the A/B
    battery per ADR-036 (``victor/evaluation/memory_ab.py``): the battery
    reports the evidence and only prompt-inflation harm demotes it.
    """

    #: Master switch for preference/episode memory. Gates BOTH the write
    #: paths and the PERCEIVE injection block. Opt-out: env
    #: ``VICTOR_MEMORY_ENABLED=false`` (env wins over settings at every read
    #: — see ``victor.agent.memory_injection.resolve_memory_enabled``).
    memory_enabled: bool = True

    #: Per-scope opt-outs (the FEP's "opt-out per scope" reads as separate
    #: flags rather than a scope enum): user-scope preferences live in the
    #: global database and project-scope preferences/episodes in the project
    #: database; a scope that is opted out is never queried and never injected.
    #: Env: VICTOR_USER_SCOPE_ENABLED / VICTOR_PROJECT_SCOPE_ENABLED.
    user_scope_enabled: bool = True
    project_scope_enabled: bool = True

    #: Hard token cap on the PERCEIVE memory block, enforced BEFORE the
    #: BudgetManager registration so memory competes for tokens instead of
    #: free-riding. Oversized store content is excluded by rank (and long
    #: entries line-truncated), never silently overflowed.
    #: Env: VICTOR_MEMORY_INJECTION_MAX_TOKENS.
    memory_injection_max_tokens: int = 500

    #: End-of-session consolidation auto-captures proposed episodes without
    #: confirmation. Default OFF — the Session-service hook proposes and
    #: stages episodes, but stores nothing until the confirm path runs
    #: (aggressive auto-capture is opt-in per FEP-0048's confirm-by-default
    #: posture). Env: VICTOR_MEMORY_AUTO_CAPTURE.
    memory_auto_capture: bool = False
