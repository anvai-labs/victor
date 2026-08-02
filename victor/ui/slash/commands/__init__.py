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

"""Slash command implementations.

This package contains all slash command implementations organized by category:

- system: help, config, status, exit, clear, theme
- session: save, load, sessions, resume, compact
- model: model, profile, provider
- tools: tools, context, lmstudio, mcp
- navigation: directory, changes, snapshots, commit
- mode: mode, build, explore, plan
- metrics: cost, metrics, serialization, learning, mlstats, bayesian
- codebase: reindex, review, init
- checkpoint: checkpoint (save, list, restore, diff, timeline)
- delegate_follow_up: delegate follow-up workflow resume
- entities: entities (list, search, show, related, stats, clear)
- debug: debug (break, clear, list, enable, disable, state, continue, step)

Commands are auto-discovered and registered when the slash module is loaded.
"""

# Import all command modules to trigger registration
from victor.ui.slash.commands import (
    bayesian,
    checkpoint,
    codebase,
    debug,
    delegate_follow_up,
    entities,
    metrics,
    mode,
    model,
    navigation,
    prompt_optimize,
    rate,
    session,
    skills,
    system,
    tools,
)

__all__ = [
    "checkpoint",
    "bayesian",
    "codebase",
    "debug",
    "delegate_follow_up",
    "entities",
    "metrics",
    "mode",
    "model",
    "navigation",
    "prompt_optimize",
    "rate",
    "session",
    "skills",
    "system",
    "tools",
]
