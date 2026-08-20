# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Structured runtime routing policy contracts.

This module defines the typed policy surface shared by batch and streaming
execution. It packages the routing-relevant signals that were previously
spread across separate hint families so callers can consume one coherent
policy object instead of stitching together ad hoc mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class StructuredRoutingPolicy:
    """Resolved routing policy for one runtime scope."""

    scope_context: Dict[str, Any] = field(default_factory=dict)
    topology_hints: Dict[str, Any] = field(default_factory=dict)
    team_hints: Dict[str, Any] = field(default_factory=dict)
    degradation_hints: Dict[str, Any] = field(default_factory=dict)
    experiment_hints: Dict[str, Any] = field(default_factory=dict)
    planning_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def selector_context(self) -> Dict[str, Any]:
        """Return the merged context used by topology and provider routing."""
        merged: Dict[str, Any] = {}
        for mapping in (
            self.topology_hints,
            self.team_hints,
            self.degradation_hints,
            self.experiment_hints,
        ):
            if isinstance(mapping, dict):
                merged.update(mapping)
        return merged

    def planning_context(self) -> Dict[str, Any]:
        """Return the planning-specific override surface."""
        return dict(self.planning_hints)

    def combined_context(self) -> Dict[str, Any]:
        """Return the merged runtime context across all policy sections."""
        merged = self.selector_context()
        merged.update(self.planning_context())
        return merged

    def is_empty(self) -> bool:
        """Return whether the policy carries any routing signal."""
        return not any(
            (
                self.topology_hints,
                self.team_hints,
                self.degradation_hints,
                self.experiment_hints,
                self.planning_hints,
                self.metadata,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the structured routing policy."""
        return {
            "scope_context": dict(self.scope_context),
            "topology_hints": dict(self.topology_hints),
            "team_hints": dict(self.team_hints),
            "degradation_hints": dict(self.degradation_hints),
            "experiment_hints": dict(self.experiment_hints),
            "planning_hints": dict(self.planning_hints),
            "selector_context": self.selector_context(),
            "combined_context": self.combined_context(),
            "metadata": dict(self.metadata),
        }


__all__ = ["StructuredRoutingPolicy"]
