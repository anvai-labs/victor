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

"""Framework-first helpers for resolving and running configured teams and workflows."""

from __future__ import annotations

import logging
from inspect import getattr_static
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from victor.framework.teams import AgentTeam
from victor.teams.types import TeamFormation, TeamResult

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTeamExecutionPlan:
    """Concrete framework team execution plan resolved from runtime state."""

    team_name: str
    display_name: str
    formation: TeamFormation
    member_count: int
    total_tool_budget: int
    max_iterations: int
    max_workers: Optional[int] = None
    recommendation_source: str = "explicit"
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _team_spec: Any = field(default=None, repr=False, compare=False)
    _members: Tuple[Any, ...] = field(default_factory=tuple, repr=False, compare=False)

    @property
    def members(self) -> Sequence[Any]:
        """Get the resolved team members for execution."""
        return self._members

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "team_name": self.team_name,
            "display_name": self.display_name,
            "formation": self.formation.value,
            "member_count": self.member_count,
            "total_tool_budget": self.total_tool_budget,
            "max_iterations": self.max_iterations,
            "max_workers": self.max_workers,
            "recommendation_source": self.recommendation_source,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    def to_runtime_context_overrides(self) -> Dict[str, Any]:
        """Convert the plan into runtime context hints."""
        overrides: Dict[str, Any] = {
            "team_name": self.team_name,
            "team_display_name": self.display_name,
            "formation_hint": self.formation.value,
        }
        if self.max_workers is not None:
            overrides["max_workers"] = self.max_workers
        return overrides


@dataclass(frozen=True)
class VerticalTeamCatalog:
    """Resolved team catalog for a framework vertical."""

    supported: bool
    provider_available: bool
    team_specs: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_team_specs(self) -> bool:
        """Whether the catalog contains configured team specs."""
        return bool(self.team_specs)

    def get(self, team_name: str) -> Any:
        """Return a named team spec when present."""
        return self.team_specs.get(team_name)

    def list_names(self) -> list[str]:
        """List configured team names."""
        return list(self.team_specs.keys())


@dataclass(frozen=True)
class VerticalWorkflowCatalog:
    """Resolved workflow catalog for a framework vertical.

    Similar to VerticalTeamCatalog, this provides a normalized interface
    for discovering workflows provided by a vertical through the canonical
    get_workflow_provider() API.
    """

    supported: bool
    provider_available: bool
    workflow_specs: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_workflow_specs(self) -> bool:
        """Whether the catalog contains workflow specs."""
        return bool(self.workflow_specs)

    def get(self, workflow_name: str) -> Any:
        """Return a named workflow spec when present."""
        return self.workflow_specs.get(workflow_name)

    def list_names(self) -> List[str]:
        """List available workflow names."""
        return list(self.workflow_specs.keys())


@dataclass(frozen=True)
class VerticalCoordinationCatalog:
    """Combined team/workflow catalog for a vertical-facing surface."""

    team_catalog: VerticalTeamCatalog = field(
        default_factory=lambda: VerticalTeamCatalog(supported=False, provider_available=False)
    )
    workflow_catalog: VerticalWorkflowCatalog = field(
        default_factory=lambda: VerticalWorkflowCatalog(supported=False, provider_available=False)
    )

    @property
    def has_team_specs(self) -> bool:
        """Whether a team catalog is populated."""
        return self.team_catalog.has_team_specs

    @property
    def has_workflow_specs(self) -> bool:
        """Whether a workflow catalog is populated."""
        return self.workflow_catalog.has_workflow_specs

    def list_team_names(self) -> List[str]:
        """List available team names."""
        return self.team_catalog.list_names()

    def list_workflow_names(self) -> List[str]:
        """List available workflow names."""
        return self.workflow_catalog.list_names()


def resolve_registered_team_catalogs() -> Dict[str, VerticalTeamCatalog]:
    """Resolve team catalogs exposed by all registered vertical providers.

    This is the shared framework-facing discovery surface for inspection paths
    such as CLI capability listings. It normalizes provider-registry payloads
    into the same ``VerticalTeamCatalog`` shape used elsewhere in the framework.
    """
    return _resolve_registered_catalogs(
        registry_method="get_all_team_specs",
        catalog_builder=lambda payload: VerticalTeamCatalog(
            supported=True,
            provider_available=bool(payload),
            team_specs=payload,
        ),
    )


def resolve_registered_workflow_catalogs() -> Dict[str, VerticalWorkflowCatalog]:
    """Resolve workflow catalogs exposed by all registered vertical providers."""
    return _resolve_registered_catalogs(
        registry_method="get_all_workflows",
        catalog_builder=lambda payload: VerticalWorkflowCatalog(
            supported=True,
            provider_available=bool(payload),
            workflow_specs=payload,
        ),
    )


def list_registered_team_names(
    *,
    vertical: Optional[str] = None,
    qualify_with_vertical: bool = True,
) -> List[str]:
    """List registered team names from the shared framework catalog surface."""
    return _list_registered_catalog_names(
        resolve_registered_team_catalogs(),
        vertical=vertical,
        qualify_with_vertical=qualify_with_vertical,
    )


def list_registered_workflow_names(
    *,
    vertical: Optional[str] = None,
    qualify_with_vertical: bool = True,
) -> List[str]:
    """List registered workflow names from the shared framework catalog surface."""
    return _list_registered_catalog_names(
        resolve_registered_workflow_catalogs(),
        vertical=vertical,
        qualify_with_vertical=qualify_with_vertical,
    )


def resolve_vertical_coordination_catalog(subject: Any) -> VerticalCoordinationCatalog:
    """Resolve both team and workflow catalogs for a single vertical surface."""
    return VerticalCoordinationCatalog(
        team_catalog=resolve_vertical_team_catalog(subject),
        workflow_catalog=resolve_vertical_workflow_catalog(subject),
    )


def resolve_registered_coordination_catalogs() -> Dict[str, VerticalCoordinationCatalog]:
    """Resolve combined team/workflow catalogs for all registered verticals."""
    team_catalogs = resolve_registered_team_catalogs()
    workflow_catalogs = resolve_registered_workflow_catalogs()
    vertical_names = set(team_catalogs) | set(workflow_catalogs)

    resolved: Dict[str, VerticalCoordinationCatalog] = {}
    for vertical_name in vertical_names:
        resolved[vertical_name] = VerticalCoordinationCatalog(
            team_catalog=team_catalogs.get(vertical_name, _empty_team_catalog()),
            workflow_catalog=workflow_catalogs.get(vertical_name, _empty_workflow_catalog()),
        )
    return resolved


def resolve_configured_team(
    orchestrator: Any,
    *,
    task_type: str,
    complexity: str,
    preferred_team: Optional[str] = None,
    preferred_formation: Optional[str] = None,
    max_workers: Optional[int] = None,
    tool_budget: Optional[int] = None,
) -> Optional[ResolvedTeamExecutionPlan]:
    """Resolve the best configured team for a runtime topology plan."""
    team_specs = _get_team_specs(orchestrator)
    if not team_specs:
        return None

    team_name = preferred_team if preferred_team in team_specs else None
    recommendation_source = "explicit" if team_name else "unresolved"
    recommendation_reason = None
    recommended_formation = None

    if team_name is None:
        suggestion = _get_coordination_suggestion(
            orchestrator,
            task_type=task_type,
            complexity=complexity,
        )
        primary_team = getattr(suggestion, "primary_team", None) if suggestion is not None else None
        candidate_name = getattr(primary_team, "team_name", None)
        if candidate_name in team_specs:
            team_name = str(candidate_name)
            recommended_formation = getattr(primary_team, "formation", None)
            recommendation_reason = getattr(primary_team, "reason", None)
            recommendation_source = getattr(primary_team, "source", "coordination")

    if team_name is None and len(team_specs) == 1:
        team_name = next(iter(team_specs))
        recommendation_source = "single_available_team"

    if team_name is None:
        return None

    team_spec = team_specs.get(team_name)
    if team_spec is None:
        return None

    formation = (
        _normalize_formation(preferred_formation)
        or _normalize_formation(recommended_formation)
        or _normalize_formation(getattr(team_spec, "formation", None))
        or TeamFormation.SEQUENTIAL
    )
    members = _limit_team_members(
        getattr(team_spec, "members", ()) or (),
        formation=formation,
        max_workers=max_workers,
    )
    total_tool_budget = _coerce_positive_int(tool_budget)
    if total_tool_budget is None:
        total_tool_budget = _coerce_positive_int(getattr(team_spec, "total_tool_budget", None)) or 0
    max_iterations = _coerce_positive_int(getattr(team_spec, "max_iterations", None)) or 50

    return ResolvedTeamExecutionPlan(
        team_name=str(team_name),
        display_name=str(getattr(team_spec, "name", team_name)),
        formation=formation,
        member_count=len(members),
        total_tool_budget=total_tool_budget,
        max_iterations=max_iterations,
        max_workers=max_workers,
        recommendation_source=recommendation_source,
        rationale=recommendation_reason,
        metadata={
            "task_type": task_type,
            "complexity": complexity,
        },
        _team_spec=team_spec,
        _members=tuple(members),
    )


async def execute_resolved_team(
    orchestrator: Any,
    *,
    goal: str,
    resolved_plan: ResolvedTeamExecutionPlan,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 600,
) -> TeamResult:
    """Execute a previously resolved configured team via framework APIs."""
    team_spec = resolved_plan._team_spec
    if team_spec is None:
        raise ValueError("Resolved team plan is missing team spec details")

    team = await AgentTeam.create(
        orchestrator=orchestrator,
        name=str(getattr(team_spec, "name", resolved_plan.display_name)),
        goal=goal,
        members=list(resolved_plan.members),
        formation=resolved_plan.formation,
        total_tool_budget=resolved_plan.total_tool_budget,
        max_iterations=resolved_plan.max_iterations,
        timeout_seconds=timeout_seconds,
        shared_context=context,
    )
    return await team.run()


async def run_configured_team(
    orchestrator: Any,
    *,
    goal: str,
    task_type: str,
    complexity: str,
    preferred_team: Optional[str] = None,
    preferred_formation: Optional[str] = None,
    max_workers: Optional[int] = None,
    tool_budget: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 600,
) -> Optional[tuple[ResolvedTeamExecutionPlan, TeamResult]]:
    """Resolve and execute the best available configured team."""
    resolved_plan = resolve_configured_team(
        orchestrator,
        task_type=task_type,
        complexity=complexity,
        preferred_team=preferred_team,
        preferred_formation=preferred_formation,
        max_workers=max_workers,
        tool_budget=tool_budget,
    )
    if resolved_plan is None:
        return None

    result = await execute_resolved_team(
        orchestrator,
        goal=goal,
        resolved_plan=resolved_plan,
        context=context,
        timeout_seconds=timeout_seconds,
    )
    return resolved_plan, result


def resolve_named_team(
    orchestrator: Any,
    *,
    team_name: str,
    preferred_formation: Optional[str] = None,
    max_workers: Optional[int] = None,
    tool_budget: Optional[int] = None,
) -> Optional[ResolvedTeamExecutionPlan]:
    """Resolve an explicitly named configured team into a concrete execution plan."""
    team_specs = _get_team_specs(orchestrator)
    if not team_specs or team_name not in team_specs:
        return None

    return resolve_configured_team(
        orchestrator,
        task_type=str(team_name),
        complexity="high",
        preferred_team=team_name,
        preferred_formation=preferred_formation,
        max_workers=max_workers,
        tool_budget=tool_budget,
    )


async def run_named_team(
    orchestrator: Any,
    *,
    team_name: str,
    goal: str,
    preferred_formation: Optional[str] = None,
    max_workers: Optional[int] = None,
    tool_budget: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 600,
) -> Optional[tuple[ResolvedTeamExecutionPlan, TeamResult]]:
    """Resolve and execute an explicitly named configured team."""
    resolved_plan = resolve_named_team(
        orchestrator,
        team_name=team_name,
        preferred_formation=preferred_formation,
        max_workers=max_workers,
        tool_budget=tool_budget,
    )
    if resolved_plan is None:
        return None

    result = await execute_resolved_team(
        orchestrator,
        goal=goal,
        resolved_plan=resolved_plan,
        context=context,
        timeout_seconds=timeout_seconds,
    )
    return resolved_plan, result


def resolve_vertical_team_catalog(vertical: Any) -> VerticalTeamCatalog:
    """Resolve team specs from a vertical or cached vertical context.

    Prefers the canonical team-spec provider API when a live vertical is
    available. If the caller only has a VerticalContext-like snapshot, falls
    back to its cached ``team_specs`` mapping so interface layers can reuse the
    same framework discovery surface.
    """
    team_specs_snapshot = _get_catalog_snapshot(vertical, "team_specs")
    supports_snapshot = _supports_catalog_snapshot(vertical, "team_specs")
    vertical_source = _resolve_vertical_catalog_source(vertical, "get_team_spec_provider")
    if vertical_source is None:
        if supports_snapshot:
            return VerticalTeamCatalog(
                supported=True,
                provider_available=bool(team_specs_snapshot),
                team_specs=team_specs_snapshot,
            )
        return VerticalTeamCatalog(supported=False, provider_available=False)

    team_provider = vertical_source.get_team_spec_provider()
    if team_provider is None:
        return VerticalTeamCatalog(
            supported=True,
            provider_available=bool(team_specs_snapshot),
            team_specs=team_specs_snapshot,
        )

    get_team_specs = getattr(team_provider, "get_team_specs", None)
    if not callable(get_team_specs):
        return VerticalTeamCatalog(
            supported=True,
            provider_available=bool(team_specs_snapshot),
            team_specs=team_specs_snapshot,
        )

    try:
        team_specs = get_team_specs()
    except Exception as exc:
        logger.debug("Failed to get team specs from provider: %s", exc)
        team_specs = None

    if isinstance(team_specs, dict):
        return VerticalTeamCatalog(
            supported=True,
            provider_available=True,
            team_specs=dict(team_specs),
        )
    return VerticalTeamCatalog(
        supported=True,
        provider_available=True,
        team_specs=team_specs_snapshot,
    )


def resolve_vertical_workflow_catalog(vertical: Any) -> VerticalWorkflowCatalog:
    """Resolve workflow specs from a vertical or cached vertical context.

    Prefers the canonical workflow provider API when available. If the caller
    only has a VerticalContext-like snapshot, falls back to its cached
    ``workflows`` mapping so chat, CLI, and framework surfaces share one
    catalog-resolution path.

    Args:
        vertical: A vertical instance, a VerticalContext-like snapshot, or any
            object that exposes the canonical workflow provider API.

    Returns:
        VerticalWorkflowCatalog with supported, provider_available, and workflow_specs

    Example:
        catalog = resolve_vertical_workflow_catalog(MyCodingVertical)
        if catalog.supported and catalog.provider_available:
            for name in catalog.list_names():
                print(f"Workflow: {name}")
    """
    workflow_specs_snapshot = _get_catalog_snapshot(vertical, "workflows")
    supports_snapshot = _supports_catalog_snapshot(vertical, "workflows")
    vertical_source = _resolve_vertical_catalog_source(vertical, "get_workflow_provider")
    if vertical_source is None:
        if supports_snapshot:
            return VerticalWorkflowCatalog(
                supported=True,
                provider_available=bool(workflow_specs_snapshot),
                workflow_specs=workflow_specs_snapshot,
            )
        return VerticalWorkflowCatalog(supported=False, provider_available=False)

    workflow_provider = vertical_source.get_workflow_provider()
    if workflow_provider is None:
        return VerticalWorkflowCatalog(
            supported=True,
            provider_available=bool(workflow_specs_snapshot),
            workflow_specs=workflow_specs_snapshot,
        )

    # Try get_workflows() first (returns Dict[str, Any])
    get_workflows = getattr(workflow_provider, "get_workflows", None)
    if callable(get_workflows):
        try:
            workflow_specs = get_workflows()
            if isinstance(workflow_specs, dict):
                return VerticalWorkflowCatalog(
                    supported=True,
                    provider_available=True,
                    workflow_specs=dict(workflow_specs),
                )
        except Exception as exc:
            logger.debug("Failed to get workflows from provider: %s", exc)

    # Fallback: try get_workflow_names() (returns List[str])
    # Build a minimal spec dict with names as keys
    get_workflow_names = getattr(workflow_provider, "get_workflow_names", None)
    if callable(get_workflow_names):
        try:
            workflow_names = get_workflow_names()
            if isinstance(workflow_names, (list, tuple)):
                return VerticalWorkflowCatalog(
                    supported=True,
                    provider_available=True,
                    workflow_specs=dict.fromkeys(workflow_names, None),
                )
        except Exception as exc:
            logger.debug("Failed to get workflow names from provider: %s", exc)

    # Provider exists but doesn't have callable methods
    return VerticalWorkflowCatalog(
        supported=True,
        provider_available=bool(workflow_specs_snapshot),
        workflow_specs=workflow_specs_snapshot,
    )


def _get_team_specs(orchestrator: Any) -> Dict[str, Any]:
    """Read configured team specs from the canonical runtime surfaces."""
    getter = getattr(orchestrator, "get_team_specs", None)
    if callable(getter):
        try:
            specs = getter()
            if isinstance(specs, dict) and specs:
                return specs
        except Exception as exc:
            logger.debug("Failed to read orchestrator team specs: %s", exc)

    get_vertical_context = getattr(orchestrator, "get_vertical_context", None)
    if callable(get_vertical_context):
        try:
            vertical_context = get_vertical_context()
        except Exception as exc:
            logger.debug("Failed to read vertical context for team specs: %s", exc)
            vertical_context = None
    else:
        vertical_context = getattr(orchestrator, "vertical_context", None)

    specs = getattr(vertical_context, "team_specs", None)
    return dict(specs) if isinstance(specs, dict) else {}


def _resolve_registered_catalogs(
    *,
    registry_method: str,
    catalog_builder: Any,
) -> Dict[str, Any]:
    """Normalize provider-registry payloads into per-vertical catalogs."""
    try:
        from victor.framework.providers.protocol import get_provider_registry

        provider_registry = get_provider_registry()
        get_all = getattr(provider_registry, registry_method, None)
        if not callable(get_all):
            return {}
        raw_catalogs = get_all()
    except Exception as exc:
        logger.debug("Failed to resolve registered catalogs via %s: %s", registry_method, exc)
        return {}

    if not isinstance(raw_catalogs, dict):
        return {}

    resolved: Dict[str, Any] = {}
    default_payload: Dict[str, Any] = {}
    for vertical_name, payload in raw_catalogs.items():
        if isinstance(payload, dict):
            resolved[str(vertical_name)] = catalog_builder(dict(payload))
        else:
            default_payload[str(vertical_name)] = payload

    if default_payload:
        resolved["default"] = catalog_builder(default_payload)

    return resolved


def _list_registered_catalog_names(
    catalogs: Dict[str, Any],
    *,
    vertical: Optional[str],
    qualify_with_vertical: bool,
) -> List[str]:
    """Flatten registered catalog names for inspection surfaces."""
    if vertical is not None:
        catalog = catalogs.get(vertical)
        if catalog is None:
            return []
        return list(catalog.list_names())

    names: List[str] = []
    for vertical_name, catalog in catalogs.items():
        for item_name in catalog.list_names():
            if qualify_with_vertical:
                names.append(f"{vertical_name}:{item_name}")
            else:
                names.append(item_name)
    return names


def _empty_team_catalog() -> VerticalTeamCatalog:
    """Create an empty team catalog."""
    return VerticalTeamCatalog(supported=False, provider_available=False)


def _empty_workflow_catalog() -> VerticalWorkflowCatalog:
    """Create an empty workflow catalog."""
    return VerticalWorkflowCatalog(supported=False, provider_available=False)


def _resolve_vertical_catalog_source(subject: Any, provider_method: str) -> Optional[Any]:
    """Resolve the canonical vertical-like object for catalog discovery."""
    if subject is None:
        return None
    if _has_declared_attribute(subject, provider_method):
        return subject
    config = (
        getattr(subject, "config", None) if _has_declared_attribute(subject, "config") else None
    )
    if config is not None and _has_declared_attribute(config, provider_method):
        return config
    return None


def _supports_catalog_snapshot(subject: Any, attr_name: str) -> bool:
    """Whether the subject exposes a cached catalog snapshot."""
    return _has_declared_attribute(subject, attr_name)


def _get_catalog_snapshot(subject: Any, attr_name: str) -> Dict[str, Any]:
    """Read a cached dict snapshot for a team/workflow catalog."""
    mapping = getattr(subject, attr_name, None)
    return dict(mapping) if isinstance(mapping, dict) else {}


def _has_declared_attribute(subject: Any, attr_name: str) -> bool:
    """Check for a real attribute without triggering MagicMock fallbacks."""
    if subject is None:
        return False
    try:
        getattr_static(subject, attr_name)
    except AttributeError:
        return False
    return True


def _get_coordination_suggestion(
    orchestrator: Any,
    *,
    task_type: str,
    complexity: str,
) -> Optional[Any]:
    """Ask the shared framework coordination engine for a team recommendation."""
    try:
        from victor.framework.coordination_runtime import (
            get_runtime_coordination_suggestion,
        )

        return get_runtime_coordination_suggestion(
            runtime_subject=orchestrator,
            task_type=task_type,
            complexity=complexity,
        )
    except Exception as exc:
        logger.debug("Failed to get coordination team suggestion: %s", exc)
        return None


def _normalize_formation(value: Any) -> Optional[TeamFormation]:
    """Coerce formation-like values into TeamFormation."""
    if value is None:
        return None
    if isinstance(value, TeamFormation):
        return value
    try:
        return TeamFormation(str(getattr(value, "value", value)))
    except ValueError:
        return None


def _limit_team_members(
    members: Sequence[Any],
    *,
    formation: TeamFormation,
    max_workers: Optional[int],
) -> Sequence[Any]:
    """Bound the concrete team to the requested worker count."""
    worker_limit = _coerce_positive_int(max_workers)
    if worker_limit is None or worker_limit >= len(members):
        return list(members)

    if formation == TeamFormation.HIERARCHICAL:
        selected = [
            member
            for member in members
            if getattr(member, "is_supervisor", False) or getattr(member, "is_manager", False)
        ]
        limited = selected[:1]
        for member in members:
            if member in limited:
                continue
            limited.append(member)
            if len(limited) >= worker_limit:
                break
        return limited

    return list(members[:worker_limit])


def _coerce_positive_int(value: Any) -> Optional[int]:
    """Convert runtime limits into positive integers when possible."""
    if value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


__all__ = [
    "ResolvedTeamExecutionPlan",
    "VerticalCoordinationCatalog",
    "VerticalTeamCatalog",
    "VerticalWorkflowCatalog",
    "execute_resolved_team",
    "list_registered_team_names",
    "list_registered_workflow_names",
    "resolve_configured_team",
    "resolve_named_team",
    "resolve_registered_coordination_catalogs",
    "resolve_registered_team_catalogs",
    "resolve_registered_workflow_catalogs",
    "resolve_vertical_coordination_catalog",
    "resolve_vertical_team_catalog",
    "resolve_vertical_workflow_catalog",
    "run_named_team",
    "run_configured_team",
]
