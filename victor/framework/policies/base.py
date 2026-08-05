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

"""Base class for governance policies.

A policy is a small, single-responsibility unit (cost budget, tool gate, …).
The shape deliberately mirrors the lightweight middleware classes in
``victor/framework/middleware.py`` (e.g. ``GitSafetyMiddleware``): declare the
phases and tools you care about, then implement :meth:`evaluate`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

from victor.framework.policies.types import Phase, PolicyEvent, PolicyVerdict


class Policy(ABC):
    """Abstract base for a single governance policy.

    Subclasses set :attr:`name` and implement :meth:`evaluate`. Override
    :meth:`phases` and :meth:`applies_to` to scope when the policy runs; the
    engine skips evaluation for events that don't match, so :meth:`evaluate`
    only ever sees relevant events.
    """

    #: Stable identifier used in audit events and elicitation messages.
    name: str = "policy"

    def phases(self) -> Set[Phase]:
        """Phases this policy participates in (default: TOOL_CALL only)."""
        return {Phase.TOOL_CALL}

    def applies_to(self, tool_name: str) -> bool:
        """Whether this policy applies to ``tool_name`` (default: all tools)."""
        return True

    @abstractmethod
    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Return a verdict for ``event``.

        Implementations must be side-effect-light and fast; they run on the
        critical path of every applicable tool call.
        """
        raise NotImplementedError


__all__ = ["Policy"]
