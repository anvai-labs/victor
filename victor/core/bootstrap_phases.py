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

"""Declarative bootstrap phase DAG for service container initialization.

Replaces the implicit ordering in bootstrap_container() with an explicit
dependency graph. Each phase declares its dependencies, and execution
follows a topological sort to ensure correct ordering.

Usage:
    from victor.core.bootstrap_phases import BOOTSTRAP_PHASES, execute_phases

    execute_phases(BOOTSTRAP_PHASES, container, settings, active_vertical)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


@dataclass
class BootstrapPhase:
    """A single phase in the bootstrap dependency graph.

    Attributes:
        name: Unique identifier for this phase.
        register_fn: Callable(container, settings, context) that performs registration.
        depends_on: Phase names that must complete before this phase runs.
        optional: If True, failure logs a warning instead of raising.
        condition: If provided, phase is skipped when condition(settings) returns False.
    """

    name: str
    register_fn: Callable[..., None]
    depends_on: tuple = ()
    optional: bool = False
    condition: Optional[Callable[..., bool]] = None


def topological_sort(phases: List[BootstrapPhase]) -> List[BootstrapPhase]:
    """Sort phases in dependency order. Raises ValueError on cycles."""
    by_name = {p.name: p for p in phases}
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    result: List[BootstrapPhase] = []

    def visit(name: str) -> None:
        if name in in_stack:
            raise ValueError(f"Cycle detected in bootstrap phases involving '{name}'")
        if name in visited:
            return
        in_stack.add(name)
        phase = by_name[name]
        for dep in phase.depends_on:
            if dep not in by_name:
                raise ValueError(f"Phase '{name}' depends on unknown phase '{dep}'")
            visit(dep)
        in_stack.remove(name)
        visited.add(name)
        result.append(phase)

    for phase in phases:
        visit(phase.name)

    return result


def execute_phases(
    phases: List[BootstrapPhase],
    container: Any,
    settings: Any,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Execute bootstrap phases in topological order.

    Args:
        phases: List of BootstrapPhase definitions.
        container: ServiceContainer instance.
        settings: Settings instance.
        context: Mutable dict for inter-phase data sharing
            (e.g., active_vertical, override_services).

    Returns:
        List of phase names that were successfully executed.
    """
    if context is None:
        context = {}

    ordered = topological_sort(phases)
    executed: List[str] = []

    for phase in ordered:
        # Check condition gate
        if phase.condition is not None:
            try:
                if not phase.condition(settings, context):
                    logger.debug("Phase '%s' skipped (condition not met)", phase.name)
                    continue
            except Exception as e:
                logger.debug("Phase '%s' condition check failed: %s", phase.name, e)
                if not phase.optional:
                    raise
                continue

        try:
            phase.register_fn(container, settings, context)
            executed.append(phase.name)
            logger.debug("Phase '%s' completed", phase.name)
        except Exception as e:
            if phase.optional:
                logger.warning("Optional phase '%s' failed: %s", phase.name, e)
            else:
                logger.error("Critical phase '%s' failed: %s", phase.name, e)
                raise

    return executed
