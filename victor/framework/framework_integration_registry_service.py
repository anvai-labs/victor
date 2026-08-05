# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework-owned facade for vertical integration registry side effects."""

from __future__ import annotations

import copy
import hashlib
from victor.core.json_utils import json_dumps
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Optional, Sequence


@dataclass
class RegistrationMetrics:
    """Metrics for registration attempts by artifact kind."""

    attempted: int = 0
    applied: int = 0
    skipped: int = 0
    failed: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "attempted": self.attempted,
            "applied": self.applied,
            "skipped": self.skipped,
            "failed": self.failed,
        }


class FrameworkIntegrationRegistryService:
    """Facade for global registry interactions during framework integration.

    This service centralizes all framework-level registration side effects
    previously scattered across FrameworkStepHandler.
    """

    def __init__(self) -> None:
        self._fingerprints: Dict[str, str] = {}
        self._metrics: Dict[str, RegistrationMetrics] = {}
        self._lock = RLock()

    def register_workflows(
        self,
        vertical_name: str,
        workflows: Dict[str, Any],
        *,
        replace: bool = True,
        registration_version: Optional[str] = None,
    ) -> int:
        """Register namespaced workflows in the global workflow registry."""
        import victor.workflows.registry as workflow_registry_module

        get_registry = getattr(workflow_registry_module, "get_workflow_registry", None)
        if not callable(get_registry):
            get_registry = workflow_registry_module.get_global_registry
        registry = get_registry()

        applied = 0
        for short_name, workflow in workflows.items():
            namespaced_name = f"{vertical_name}:{short_name}"
            workflow_obj, restore = self._workflow_with_name(workflow, namespaced_name)
            scope_key = f"{vertical_name}:{short_name}"
            fingerprint = self._fingerprint_value(workflow_obj)

            if not self._should_apply(
                "workflows",
                scope_key,
                fingerprint,
                registration_version=registration_version,
            ):
                if restore is not None:
                    restore()
                continue

            try:
                registry.register(workflow_obj, replace=replace)
                applied += 1
                self._record_applied("workflows", 1)
            except Exception:
                self._record_failure("workflows")
                raise
            finally:
                if restore is not None:
                    restore()

        return applied

    def register_workflow_triggers(
        self,
        vertical_name: str,
        auto_workflows: Sequence[Any],
        *,
        registration_version: Optional[str] = None,
    ) -> int:
        """Register workflow triggers for a vertical."""
        payload = list(auto_workflows)
        fingerprint = self._fingerprint_value(payload)
        if not self._should_apply(
            "workflow_triggers",
            vertical_name,
            fingerprint,
            registration_version=registration_version,
        ):
            return 0

        from victor.workflows.trigger_registry import get_trigger_registry

        registry = get_trigger_registry()
        try:
            registry.register_from_vertical(vertical_name, payload)
            self._record_applied("workflow_triggers", len(payload))
            return len(payload)
        except Exception:
            self._record_failure("workflow_triggers")
            raise

    def set_active_rl_learners(
        self,
        learner_names: Sequence[str],
        *,
        registration_version: Optional[str] = None,
    ) -> None:
        """Configure global RL coordinator active learners."""
        payload = list(learner_names)
        fingerprint = self._fingerprint_value(payload)
        if not self._should_apply(
            "rl_learners",
            "global",
            fingerprint,
            registration_version=registration_version,
        ):
            return

        from victor.framework.rl.coordinator import get_rl_coordinator

        coordinator = get_rl_coordinator()
        try:
            coordinator.set_active_learners(payload)
            self._record_applied("rl_learners", 1)
        except Exception:
            self._record_failure("rl_learners")
            raise

    def register_team_specs(
        self,
        vertical_name: str,
        team_specs: Dict[str, Any],
        *,
        replace: bool = True,
        registration_version: Optional[str] = None,
    ) -> int:
        """Register team specs in TeamSpecRegistry."""
        fingerprint = self._fingerprint_value(team_specs)
        if not self._should_apply(
            "team_specs",
            vertical_name,
            fingerprint,
            registration_version=registration_version,
        ):
            return 0

        from victor.framework.team_registry import get_team_registry

        registry = get_team_registry()
        try:
            count = registry.register_from_vertical(vertical_name, team_specs, replace=replace)
            self._record_applied("team_specs", count)
            return count
        except Exception:
            self._record_failure("team_specs")
            raise

    def register_chains(
        self,
        vertical_name: str,
        chains: Dict[str, Any],
        *,
        replace: bool = True,
        registration_version: Optional[str] = None,
    ) -> int:
        """Register chains in ChainRegistry."""
        fingerprint = self._fingerprint_value(chains)
        if not self._should_apply(
            "chains",
            vertical_name,
            fingerprint,
            registration_version=registration_version,
        ):
            return 0

        from victor.framework.chain_registry import get_chain_registry

        registry = get_chain_registry()
        try:
            count = registry.register_from_vertical(vertical_name, chains, replace=replace)
            self._record_applied("chains", count)
            return count
        except Exception:
            self._record_failure("chains")
            raise

    def register_personas(
        self,
        vertical_name: str,
        personas: Dict[str, Any],
        *,
        replace: bool = True,
        registration_version: Optional[str] = None,
    ) -> int:
        """Register personas in PersonaRegistry."""
        fingerprint = self._fingerprint_value(personas)
        if not self._should_apply(
            "personas",
            vertical_name,
            fingerprint,
            registration_version=registration_version,
        ):
            return 0

        from victor.framework.persona_registry import get_persona_registry

        registry = get_persona_registry()
        try:
            count = registry.register_from_vertical(vertical_name, personas, replace=replace)
            self._record_applied("personas", count)
            return count
        except Exception:
            self._record_failure("personas")
            raise

    def register_tool_graph(
        self,
        vertical_name: str,
        graph: Any,
        *,
        registration_version: Optional[str] = None,
    ) -> None:
        """Register a tool graph for a vertical."""
        fingerprint = self._fingerprint_value(graph)
        if not self._should_apply(
            "tool_graph",
            vertical_name,
            fingerprint,
            registration_version=registration_version,
        ):
            return

        from victor.tools.tool_graph import ToolGraphRegistry

        registry = ToolGraphRegistry.get_instance()
        try:
            registry.register_graph(vertical_name, graph)
            self._record_applied("tool_graph", 1)
        except Exception:
            self._record_failure("tool_graph")
            raise

    def register_handlers(
        self,
        vertical_name: str,
        handlers: Dict[str, Any],
        *,
        replace: bool = True,
        registration_version: Optional[str] = None,
    ) -> int:
        """Register handlers through framework registry with executor fallback."""
        fingerprint = self._fingerprint_value(handlers)
        if not self._should_apply(
            "handlers",
            vertical_name,
            fingerprint,
            registration_version=registration_version,
        ):
            return 0

        try:
            from victor.framework.handler_registry import get_handler_registry

            registry = get_handler_registry()
            if hasattr(registry, "register_vertical") and callable(registry.register_vertical):
                registry.register_vertical(vertical_name, handlers)
            elif hasattr(registry, "register") and callable(registry.register):
                for name, handler in handlers.items():
                    registry.register(name, handler, vertical=vertical_name, replace=replace)
            else:
                raise AttributeError("HandlerRegistry has no supported registration method")

            self._record_applied("handlers", len(handlers))
            return len(handlers)
        except ImportError:
            from victor.workflows.compute_registry import register_compute_handler

            for name, handler in handlers.items():
                register_compute_handler(name, handler)
            self._record_applied("handlers", len(handlers))
            return len(handlers)
        except Exception:
            self._record_failure("handlers")
            raise

    def snapshot_metrics(self) -> Dict[str, Dict[str, int]]:
        """Return copy of registration metrics by artifact kind."""
        with self._lock:
            return {kind: metrics.to_dict() for kind, metrics in self._metrics.items()}

    def clear_registration_state(self) -> None:
        """Clear idempotence fingerprints and metrics (testing/maintenance helper)."""
        with self._lock:
            self._fingerprints.clear()
            self._metrics.clear()

    def _workflow_with_name(
        self,
        workflow: Any,
        target_name: str,
    ) -> tuple[Any, Optional[Callable[[], None]]]:
        """Return workflow object with target name and optional restore callback."""
        if not hasattr(workflow, "name"):
            return workflow, None

        current_name = getattr(workflow, "name", None)
        if current_name == target_name:
            return workflow, None

        for copier in (copy.deepcopy, copy.copy):
            try:
                clone = copier(workflow)
                clone.name = target_name
                return clone, None
            except Exception:
                continue

        workflow.name = target_name

        def _restore() -> None:
            workflow.name = current_name

        return workflow, _restore

    def _should_apply(
        self,
        kind: str,
        scope_key: str,
        fingerprint: str,
        *,
        registration_version: Optional[str] = None,
    ) -> bool:
        """Return True when registration should execute for this fingerprint."""
        composite_key = self._composite_scope_key(
            kind,
            scope_key,
            registration_version=registration_version,
        )
        with self._lock:
            metrics = self._metrics.setdefault(kind, RegistrationMetrics())
            metrics.attempted += 1
            if self._fingerprints.get(composite_key) == fingerprint:
                metrics.skipped += 1
                return False
            self._fingerprints[composite_key] = fingerprint
            return True

    def _composite_scope_key(
        self,
        kind: str,
        scope_key: str,
        *,
        registration_version: Optional[str] = None,
    ) -> str:
        """Compose dedupe key for a registration scope and optional version token."""
        token = str(registration_version).strip() if registration_version is not None else ""
        if token:
            return f"{kind}:{scope_key}@{token}"
        return f"{kind}:{scope_key}"

    def _record_applied(self, kind: str, count: int) -> None:
        with self._lock:
            metrics = self._metrics.setdefault(kind, RegistrationMetrics())
            metrics.applied += max(0, int(count))

    def _record_failure(self, kind: str) -> None:
        with self._lock:
            metrics = self._metrics.setdefault(kind, RegistrationMetrics())
            metrics.failed += 1

    def _fingerprint_value(self, value: Any) -> str:
        """Create a stable fingerprint for dedupe comparisons."""
        normalized = self._normalize_for_fingerprint(value)
        payload = json_dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _normalize_for_fingerprint(self, value: Any) -> Any:
        """Normalize values to JSON-friendly deterministic form."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda kv: str(kv[0]))
            return {str(k): self._normalize_for_fingerprint(v) for k, v in items}

        if isinstance(value, (list, tuple)):
            return [self._normalize_for_fingerprint(v) for v in value]

        if isinstance(value, set):
            normalized = [self._normalize_for_fingerprint(v) for v in value]
            return sorted(normalized, key=lambda v: repr(v))

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return self._normalize_for_fingerprint(to_dict())
            except Exception:
                pass

        if callable(value):
            module = getattr(value, "__module__", "")
            qualname = getattr(
                value, "__qualname__", getattr(value, "__name__", type(value).__name__)
            )
            return {"__callable__": f"{module}.{qualname}"}

        if hasattr(value, "__dict__"):
            attrs = {k: v for k, v in vars(value).items() if not str(k).startswith("_")}
            if attrs:
                return {
                    "__type__": type(value).__name__,
                    "__attrs__": self._normalize_for_fingerprint(attrs),
                }

        if hasattr(value, "name"):
            return {
                "__type__": type(value).__name__,
                "name": str(value.name),
            }

        return repr(value)


_registry_service_instance: Optional[FrameworkIntegrationRegistryService] = None


def get_framework_integration_registry_service() -> FrameworkIntegrationRegistryService:
    """Return singleton integration registry service."""
    global _registry_service_instance
    if _registry_service_instance is None:
        _registry_service_instance = FrameworkIntegrationRegistryService()
    return _registry_service_instance


def resolve_framework_integration_registry_service(
    orchestrator: Any,
) -> FrameworkIntegrationRegistryService:
    """Resolve integration registry service from orchestrator DI container when possible."""
    container = None

    getter = getattr(orchestrator, "get_service_container", None)
    if callable(getter):
        try:
            container = getter()
        except Exception:
            container = None

    if container is not None and hasattr(container, "get_optional"):
        try:
            service = container.get_optional(FrameworkIntegrationRegistryService)
        except Exception:
            service = None
        if isinstance(service, FrameworkIntegrationRegistryService):
            return service

        if hasattr(container, "register_instance"):
            fallback = get_framework_integration_registry_service()
            container.register_instance(FrameworkIntegrationRegistryService, fallback)
            return fallback

    return get_framework_integration_registry_service()


__all__ = [
    "FrameworkIntegrationRegistryService",
    "RegistrationMetrics",
    "get_framework_integration_registry_service",
    "resolve_framework_integration_registry_service",
]
