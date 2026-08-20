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

"""Unified event taxonomy registry.

Maps domain-specific event type enums (WorkflowEventType, TeamEventType,
AuditEventType, etc.) to the canonical ``EventType`` from
``victor.framework.events`` without merging the enums themselves.

This enables:
- Discovery of all event types across domains
- Correlation between domain events and canonical types
- Reverse lookup: which domain events map to a given canonical type

Usage::

    from victor.core.events.taxonomy import EventTaxonomyRegistry
    from victor.framework.events import EventType

    EventTaxonomyRegistry.register_domain(
        "workflow",
        WorkflowEventType,
        {
            WorkflowEventType.AGENT_CONTENT: EventType.CONTENT,
            WorkflowEventType.AGENT_ERROR: EventType.ERROR,
        },
    )

    canonical = EventTaxonomyRegistry.to_canonical(
        "workflow", WorkflowEventType.AGENT_CONTENT
    )  # → EventType.CONTENT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple, Type


@dataclass
class DomainEventMapping:
    """Mapping from a domain's event enum to canonical EventType."""

    domain: str
    enum_type: Type
    to_canonical: Dict[Any, Any] = field(default_factory=dict)


class EventTaxonomyRegistry:
    """Registry mapping domain-specific event types to canonical EventType.

    Thread-safe for reads; registration is expected at module load time.
    Uses class-level state to act as a singleton without instantiation.
    """

    _domains: ClassVar[Dict[str, DomainEventMapping]] = {}

    @classmethod
    def register_domain(
        cls,
        domain: str,
        enum_type: Type,
        mapping: Dict[Any, Any],
    ) -> None:
        """Register a domain's event type enum with its canonical mapping.

        Args:
            domain: Domain name (e.g., "workflow", "team", "audit")
            enum_type: The domain's event type enum class
            mapping: Dict mapping domain enum values to canonical EventType values
        """
        cls._domains[domain] = DomainEventMapping(
            domain=domain,
            enum_type=enum_type,
            to_canonical=dict(mapping),
        )

    @classmethod
    def to_canonical(cls, domain: str, event: Any) -> Any:
        """Map a domain event to its canonical EventType.

        Args:
            domain: Domain name
            event: Domain-specific event value

        Returns:
            Canonical EventType value, or EventType.CUSTOM if unmapped
        """
        from victor.framework.events import EventType

        domain_mapping = cls._domains.get(domain)
        if domain_mapping is None:
            return EventType.CUSTOM
        return domain_mapping.to_canonical.get(event, EventType.CUSTOM)

    @classmethod
    def from_canonical(cls, canonical: Any) -> List[Tuple[str, Any]]:
        """Reverse lookup: find all domain events mapping to a canonical type.

        Args:
            canonical: Canonical EventType value

        Returns:
            List of (domain, domain_event) tuples
        """
        results: List[Tuple[str, Any]] = []
        for domain_name, mapping in cls._domains.items():
            for domain_event, canon in mapping.to_canonical.items():
                if canon == canonical:
                    results.append((domain_name, domain_event))
        return results

    @classmethod
    def list_domains(cls) -> List[str]:
        """List all registered domain names."""
        return list(cls._domains.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered domains. Primarily for test cleanup."""
        cls._domains.clear()
