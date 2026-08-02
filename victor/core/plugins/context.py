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

"""Host implementation of PluginContext for Victor framework.

This module provides the bridge between the SDK-defined PluginContext
and the framework's internal registries and containers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set, Type

import typer
from victor_contracts import PluginContext

logger = logging.getLogger(__name__)

# EP scan cache — avoids redundant entry point scanning per vertical
_EP_SCAN_CACHE: Set[str] = set()


class HostPluginContext(PluginContext):
    """Host implementation of PluginContext.

    Wraps the framework's internal registries to provide a clean interface
    for plugins.
    """

    def __init__(self, container: Optional[Any] = None) -> None:
        """Initialize context.

        Args:
            container: The host service container.
        """
        self._container = container
        self._commands: Dict[str, typer.Typer] = {}
        # CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
        # Pending-registration buffers for surfaces that depend on bootstrap order.
        # Drained by bootstrap phases once their target registries exist.
        self._pending_rl_configs: Dict[str, Any] = {}
        self._pending_bootstrap_services: list = []
        self._pending_mcp_servers: list = []

    @property
    def commands(self) -> Dict[str, typer.Typer]:
        """Return registered commands."""
        return self._commands

    @property
    def pending_rl_configs(self) -> Dict[str, Any]:
        """Return RL configs registered by plugins via register_rl_config()."""
        return self._pending_rl_configs

    @property
    def pending_bootstrap_services(self) -> list:
        """Return bootstrap-service hooks registered by plugins."""
        return list(self._pending_bootstrap_services)

    @property
    def pending_mcp_servers(self) -> list:
        """Return MCP server specs registered by plugins."""
        return list(self._pending_mcp_servers)

    def _ensure_container(self) -> Optional[Any]:
        """Resolve and cache the host service container."""

        if self._container is None:
            from victor.core.container import get_container

            self._container = get_container()
        return self._container

    def _get_optional_service(self, *service_types: Type[Any]) -> Optional[Any]:
        """Resolve the first available service for the given protocol types."""

        container = self._ensure_container()
        if container is None:
            return None

        for service_type in service_types:
            try:
                if hasattr(container, "get_optional"):
                    service = container.get_optional(service_type)
                else:
                    service = container.get(service_type)
                if service is not None:
                    return service
            except Exception:
                continue
        return None

    def register_tool(self, tool_instance: Any) -> None:
        """Register a tool with the framework's ToolRegistry."""
        try:
            registry = None

            if not self._container:
                from victor.core.container import get_container

                self._container = get_container()

            if self._container is not None:
                try:
                    from victor.agent.protocols import ToolRegistryProtocol

                    if hasattr(self._container, "get_optional"):
                        registry = self._container.get_optional(ToolRegistryProtocol)
                    elif hasattr(
                        self._container, "is_registered"
                    ) and self._container.is_registered(ToolRegistryProtocol):
                        registry = self._container.get(ToolRegistryProtocol)
                except Exception as e:
                    logger.debug(
                        "Plugin failed to resolve live ToolRegistry from container: %s",
                        e,
                    )

            if registry is None:
                from victor.tools.registry import ToolRegistry

                registry = ToolRegistry()

            if hasattr(registry, "register_tool"):
                registry.register_tool(tool_instance)
            else:
                registry.register(tool_instance)
            logger.debug(f"Plugin registered tool: {getattr(tool_instance, 'name', tool_instance)}")
        except Exception as e:
            logger.error(f"Plugin failed to register tool: {e}")

    def register_vertical(self, vertical_class: Type[Any]) -> None:
        """Register a vertical and auto-extract its capabilities.

        In addition to placing the class in VerticalRegistry, this method
        checks if the vertical declares capabilities via
        get_capability_registrations() and registers them with the
        CapabilityRegistry. This makes the plugin system the single
        authority for capability registration.
        """
        from victor.core.verticals.base import VerticalRegistry

        try:
            VerticalRegistry.register(vertical_class)
            logger.debug(
                "Plugin registered vertical: %s",
                getattr(vertical_class, "name", vertical_class),
            )
        except Exception as e:
            logger.error("Plugin failed to register vertical: %s", e)
            return

        # Auto-extract capabilities from the vertical
        self._auto_register_vertical_capabilities(vertical_class)

    def _auto_register_vertical_capabilities(self, vertical_class: Type[Any]) -> None:
        """Discover and register capabilities declared by the vertical.

        Strategy 1: Check for get_capability_registrations() class method
        that returns (protocol_type, provider) pairs.

        Strategy 2: Scan semantic extension and legacy SDK capability entry
        point groups
        for entries matching the vertical name.
        """
        vertical_name = getattr(vertical_class, "name", "")
        if not vertical_name and hasattr(vertical_class, "get_name"):
            try:
                vertical_name = vertical_class.get_name()
            except Exception:
                pass

        # Strategy 1: Explicit capability registrations
        cap_method = getattr(vertical_class, "get_capability_registrations", None)
        if cap_method and callable(cap_method):
            try:
                registrations = cap_method()
                for protocol_type, provider in registrations:
                    self.register_capability(protocol_type, provider)
                if registrations:
                    logger.debug(
                        "Auto-registered %d capabilities from %s.get_capability_registrations()",
                        len(registrations),
                        vertical_name,
                    )
            except Exception as e:
                logger.debug("Capability extraction failed for %s: %s", vertical_name, e)

        # Strategy 2: Entry point scanning (deferred — runs if vertical has EP registrations)
        # Cached to avoid redundant scanning when multiple verticals register
        if vertical_name:
            try:
                from victor.framework.entry_point_registry import (
                    CAPABILITY_ENTRY_POINT_GROUPS,
                    get_entry_point_registry,
                )

                ep_registry = get_entry_point_registry()
                for group in CAPABILITY_ENTRY_POINT_GROUPS:
                    cache_key = f"{group}:{vertical_name}"
                    if cache_key in _EP_SCAN_CACHE:
                        continue
                    _EP_SCAN_CACHE.add(cache_key)

                    group_obj = ep_registry.get_group(group)
                    if not group_obj:
                        continue
                    for ep_name, entry in group_obj.entry_points.items():
                        if ep_name.startswith(f"{vertical_name}-"):
                            try:
                                ep, loaded = entry[0], entry[1]
                                if not loaded:
                                    register_func = ep.load()
                                else:
                                    register_func = entry[2] if len(entry) > 2 else ep.load()
                                from victor.core.capability_registry import (
                                    CapabilityRegistry,
                                )

                                register_func(CapabilityRegistry.get_instance())
                                logger.debug(
                                    "Vertical %s registered EP capability: %s",
                                    vertical_name,
                                    ep_name,
                                )
                            except Exception as e:
                                logger.debug(
                                    "Skipped EP capability %s:%s: %s",
                                    group,
                                    ep_name,
                                    e,
                                )
            except Exception as e:
                logger.debug("Entry point discovery failed for %s: %s", vertical_name, e)

    def register_capability(
        self,
        protocol_type: Type[Any],
        provider: Any,
        *,
        lazy: bool = False,
    ) -> None:
        """Register a capability provider with CapabilityRegistry.

        This is the canonical way for plugins to register capabilities.
        Enhanced providers won't be downgraded by subsequent STUB registrations.
        """
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus

        registry = CapabilityRegistry.get_instance()
        if lazy and callable(provider):
            proxy = _LazyCapabilityProxy(provider)
            registry.register(protocol_type, proxy, CapabilityStatus.ENHANCED)
        else:
            registry.register(protocol_type, provider, CapabilityStatus.ENHANCED)
        logger.debug(
            "Plugin registered capability: %s (lazy=%s)",
            getattr(protocol_type, "__name__", protocol_type),
            lazy,
        )

    def register_chunker(self, chunker_instance: Any) -> None:
        """Register a chunker with the framework's ChunkingRegistry.

        .. deprecated:: 0.7.0
            Use ``register_capability(ChunkerProtocol, chunker_instance)`` instead.
            Will be removed in v1.0.
        """
        import warnings

        warnings.warn(
            "PluginContext.register_chunker() is deprecated and will be removed in v1.0. "
            "Use register_capability(ChunkerProtocol, chunker_instance) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.core.chunking.registry import get_chunking_registry

        try:
            registry = get_chunking_registry()
            registry.register(chunker_instance)
            logger.debug(
                f"Plugin registered chunker: {getattr(chunker_instance, 'name', chunker_instance)}"
            )
        except Exception as e:
            logger.error(f"Plugin failed to register chunker: {e}")

    def register_category(
        self,
        name: str,
        tools: set[str],
        *,
        description: Optional[str] = None,
    ) -> None:
        """Register a custom tool category for plugin-owned tools.

        .. deprecated:: 0.7.0
            Tool categories have no active callers. Will be removed in v1.0.
        """
        import warnings

        warnings.warn(
            "PluginContext.register_category() is deprecated and will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.framework.tools import get_category_registry

        get_category_registry().register_category(name, set(tools), description=description)
        logger.debug("Plugin registered tool category: %s", name)

    def extend_category(self, name: str, tools: set[str]) -> None:
        """Extend an existing tool category with plugin-owned tools.

        .. deprecated:: 0.7.0
            Tool categories have no active callers. Will be removed in v1.0.
        """
        import warnings

        warnings.warn(
            "PluginContext.extend_category() is deprecated and will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.framework.tools import get_category_registry

        get_category_registry().extend_category(name, set(tools))
        logger.debug("Plugin extended tool category: %s", name)

    def register_command(self, name: str, app: typer.Typer) -> None:
        """Register a command app."""
        self._commands[name] = app
        logger.debug(f"Plugin registered command: {name}")

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom workflow node executor.

        .. deprecated:: 0.7.0
            Use the ``victor.workflow_executors`` entry point group instead.
            Will be removed in v1.0.
        """
        import warnings

        warnings.warn(
            "PluginContext.register_workflow_node_executor() is deprecated and will be removed in v1.0. "
            "Use the victor.workflow_executors entry point group instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.workflows.executors.registry import register_workflow_node_executor

        register_workflow_node_executor(
            node_type,
            executor_factory,
            replace=replace,
        )

        if self._container is not None:
            try:
                from victor.workflows.compiler_protocols import (
                    NodeExecutorFactoryProtocol,
                )

                if hasattr(self._container, "is_registered") and self._container.is_registered(
                    NodeExecutorFactoryProtocol
                ):
                    factory = self._container.get_optional(NodeExecutorFactoryProtocol)
                    if factory is not None:
                        existing = getattr(factory, "_executor_types", {}).get(node_type)
                        if existing is not executor_factory:
                            factory.register_executor_type(
                                node_type,
                                executor_factory,
                                replace=replace,
                            )
            except Exception as e:
                logger.debug(f"Plugin deferred live workflow executor registration: {e}")

        logger.debug("Plugin registered workflow node executor: %s", node_type)

    def get_service(self, service_type: Type[Any]) -> Optional[Any]:
        """Retrieve a service from the container."""
        return self._get_optional_service(service_type)

    def get_provider_registry(self) -> Optional[Any]:
        """Retrieve the host's LLM provider registry via a stable SDK seam."""

        from victor_contracts.verticals.protocols import (
            ProviderRegistryProtocol as SdkProviderRegistry,
        )

        service = self._get_optional_service(SdkProviderRegistry)
        if service is not None:
            return service

        try:
            from victor.agent.protocols import (
                ProviderRegistryProtocol as HostProviderRegistry,
            )
        except ImportError:
            return None

        service = self._get_optional_service(HostProviderRegistry)
        if service is None:
            return None
        if hasattr(service, "get_provider"):
            return service
        if hasattr(service, "get") and hasattr(service, "list_providers"):
            return _ProviderRegistryAdapter(service)
        return None

    def get_graph_store(self) -> Optional[Any]:
        """Retrieve the host's graph-store service via a stable SDK seam."""

        from victor_contracts.verticals.protocols import (
            GraphStoreProtocol as SdkGraphStore,
        )

        service = self._get_optional_service(SdkGraphStore)
        if service is not None:
            return service

        try:
            from victor.storage.graph.protocol import (
                GraphStoreProtocol as HostGraphStore,
            )
        except ImportError:
            return None

        return self._get_optional_service(HostGraphStore)

    def get_vector_store(self) -> Optional[Any]:
        """Retrieve the host's vector-store service via a stable SDK seam."""

        from victor_contracts.verticals.protocols import (
            VectorStoreProtocol as SdkVectorStore,
        )

        service = self._get_optional_service(SdkVectorStore)
        if service is not None:
            return service

        try:
            from victor.framework.vertical_protocols import (
                VectorStoreProtocol as HostVectorStore,
            )
        except ImportError:
            return None

        service = self._get_optional_service(HostVectorStore)
        if service is None:
            return None
        if hasattr(service, "index_document"):
            return service
        if all(hasattr(service, attr) for attr in ("add_documents", "search", "delete")):
            return _VectorStoreAdapter(service)
        return None

    def get_embedding_service(self) -> Optional[Any]:
        """Retrieve the host's embedding service via a stable SDK seam."""

        from victor_contracts.verticals.protocols import (
            EmbeddingServiceProtocol as SdkEmbeddingService,
        )

        service = self._get_optional_service(SdkEmbeddingService)
        if service is not None:
            return service

        try:
            from victor.core.protocols import (
                EmbeddingServiceProtocol as HostEmbeddingService,
            )
        except ImportError:
            return None

        return self._get_optional_service(HostEmbeddingService)

    def get_memory_coordinator(self) -> Optional[Any]:
        """Retrieve the host's shared memory coordinator via a stable SDK seam."""

        from victor_contracts.verticals.protocols import (
            MemoryCoordinatorProtocol as SdkMemoryCoordinator,
        )

        service = self._get_optional_service(SdkMemoryCoordinator)
        if service is not None:
            return service

        try:
            from victor.agent.protocols import (
                UnifiedMemoryCoordinatorProtocol as HostMemoryCoordinator,
            )

            service = self._get_optional_service(HostMemoryCoordinator)
        except ImportError:
            service = None
        if service is not None:
            return service

        try:
            from victor.storage.memory import get_memory_coordinator

            return get_memory_coordinator()
        except Exception:
            return None

    def get_settings(self) -> Any:
        """Retrieve the application settings from the container."""
        from victor.config.settings import Settings

        settings = self._get_optional_service(Settings)
        if settings is not None:
            return settings
        else:
            # Fallback: load default settings
            return Settings()

    # CONSOLIDATION: plugin-vertical unification — see memory plugin_vertical_consolidation.md
    # Gap-fill registrations: let a plugin self-register everything via
    # register(context) instead of declaring sidecar entry-point groups.
    # The sidecar groups (victor.safety_rules, victor.tool_dependencies,
    # victor.escape_hatches, victor.rl_configs, victor.bootstrap_services,
    # victor.mcp_servers) remain readable for external packages that
    # haven't migrated.
    def register_safety_rule(self, rule: Any) -> None:
        """Register a single safety rule with the framework's SafetyEnforcer."""
        try:
            if self._container is None:
                from victor.core.container import get_container

                self._container = get_container()
            enforcer = None
            try:
                from victor.framework.config import SafetyEnforcer

                if hasattr(self._container, "is_registered") and self._container.is_registered(
                    SafetyEnforcer
                ):
                    enforcer = self._container.get(SafetyEnforcer)
            except Exception:
                enforcer = None
            if enforcer is None:
                logger.debug(
                    "SafetyEnforcer not yet registered; deferring rule %s",
                    getattr(rule, "name", rule),
                )
                return
            # SafetyEnforcer exposes add_rule; coordinators expose register_rule.
            if hasattr(enforcer, "add_rule"):
                enforcer.add_rule(rule)
            elif hasattr(enforcer, "register_rule"):
                enforcer.register_rule(rule)
            else:
                logger.debug("SafetyEnforcer has neither add_rule nor register_rule")
                return
            logger.debug("Plugin registered safety rule: %s", getattr(rule, "name", rule))
        except Exception as exc:
            logger.error("Plugin failed to register safety rule: %s", exc)

    def register_tool_dependency(self, name: str, provider: Any) -> None:
        """Register a tool dependency provider for a named vertical / tool group."""
        try:
            from victor.core.tool_dependency_loader import (
                register_vertical_tool_dependency_provider,
            )
        except ImportError:
            logger.debug("tool_dependency_loader does not expose registration helper")
            return
        try:
            register_vertical_tool_dependency_provider(name, provider)
            logger.debug("Plugin registered tool dependency: %s", name)
        except Exception as exc:
            logger.error("Plugin failed to register tool dependency %s: %s", name, exc)

    def register_escape_hatch(self, hatch: Any) -> None:
        """Register an escape hatch (condition or transform).

        ``hatch`` must expose a ``kind`` attribute (``"condition"`` or
        ``"transform"``) and ``name`` / ``fn`` attributes. Dicts with those
        keys also work.
        """
        try:
            from victor.framework.escape_hatch_registry import get_escape_hatch_registry
        except ImportError:
            logger.debug("Escape hatch registry unavailable")
            return

        def _attr(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        kind = _attr(hatch, "kind")
        name = _attr(hatch, "name")
        fn = _attr(hatch, "fn")
        if not kind or not name or fn is None:
            logger.debug("register_escape_hatch missing kind/name/fn: %s", hatch)
            return
        try:
            registry = get_escape_hatch_registry()
            if kind == "condition":
                registry.register_condition(name, fn, **(_attr(hatch, "options", {}) or {}))
            elif kind == "transform":
                registry.register_transform(name, fn, **(_attr(hatch, "options", {}) or {}))
            else:
                logger.debug("register_escape_hatch unknown kind: %s", kind)
                return
            logger.debug("Plugin registered escape hatch: %s (%s)", name, kind)
        except Exception as exc:
            logger.error("Plugin failed to register escape hatch %s: %s", name, exc)

    def register_rl_config(self, key: str, config: Any) -> None:
        """Buffer an RL config fragment; bootstrap drains it once RL services exist."""
        self._pending_rl_configs[key] = config
        logger.debug("Plugin buffered RL config: %s", key)

    def register_bootstrap_service(
        self,
        factory: Any,
        *,
        phase: str = "vertical_services",
    ) -> None:
        """Buffer a bootstrap-service hook for the named phase."""
        self._pending_bootstrap_services.append((phase, factory))
        logger.debug(
            "Plugin buffered bootstrap service for phase '%s': %s",
            phase,
            getattr(factory, "__name__", factory),
        )

    def register_mcp_server(self, spec: Any) -> None:
        """Buffer an MCP server spec for the MCP vertical to pick up."""
        self._pending_mcp_servers.append(spec)
        logger.debug("Plugin buffered MCP server spec: %s", getattr(spec, "name", spec))

    def register_domain_keyword_provider(self, provider: Any) -> None:
        """Register a domain keyword provider for adaptive compaction topic detection.

        Plugins and verticals call this to extend the topic-switch detector in
        ``AdaptiveCompactionThreshold`` with domain-specific vocabulary without
        modifying core framework code (OCP compliance).

        The provider must implement ``DomainKeywordsProviderProtocol``::

            class SwiftKeywordsProvider:
                def get_domain_keywords(self):
                    return {"swift": ["swift", "swiftui", "xcode", "cocoa"]}
                def get_provider_name(self):
                    return "swift-keywords"

            context.register_domain_keyword_provider(SwiftKeywordsProvider())

        Args:
            provider: Instance implementing ``DomainKeywordsProviderProtocol``
        """
        from victor.framework.protocols import DomainKeywordsProviderProtocol

        self.register_capability(DomainKeywordsProviderProtocol, provider)
        logger.debug(
            "Plugin registered domain keyword provider: %s",
            getattr(provider, "get_provider_name", lambda: repr(provider))(),
        )

    def register_ccg_builder(
        self,
        language: str,
        builder: Any,
    ) -> None:
        """Register a language-specific CCG builder with the capability registry.

        This allows external packages (victor-coding) to provide enhanced
        CCG construction for specific languages.

        Args:
            language: Language identifier (e.g., "python", "javascript")
            builder: CCG builder instance implementing CCGBuilderProtocol

        Example:
            context.register_ccg_builder("python", PythonCCGBuilder(graph_store))
        """
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus
        from victor.framework.vertical_protocols import CCGBuilderProtocol

        try:
            registry = CapabilityRegistry.get_instance()
            registry.register(
                CCGBuilderProtocol,
                builder,
                CapabilityStatus.ENHANCED,
                metadata={"language": language},
            )
            logger.debug(
                "Plugin registered CCG builder for language: %s",
                language,
            )
        except Exception as exc:
            logger.error("Plugin failed to register CCG builder for %s: %s", language, exc)


class _LazyCapabilityProxy:
    """Proxy that defers provider instantiation until first attribute access.

    Used by register_capability(lazy=True) to avoid importing heavy
    dependencies at plugin registration time.
    """

    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_resolved", False)

    def _resolve(self):
        """Resolve the lazy proxy by calling the factory."""
        if not object.__getattribute__(self, "_resolved"):
            factory = object.__getattribute__(self, "_factory")
            instance = factory()
            object.__setattr__(self, "_instance", instance)
            object.__setattr__(self, "_resolved", True)
        return object.__getattribute__(self, "_instance")

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to the resolved instance."""
        if name in ("_factory", "_instance", "_resolved"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._resolve(), name, value)

    def __call__(self, *args, **kwargs):
        """Support calling the proxy as a constructor/factory."""
        resolved = self._resolve()
        return resolved(*args, **kwargs)


class _ProviderRegistryAdapter:
    """Adapter exposing host provider registries through the SDK protocol."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def get_provider(self, name: str) -> Any:
        getter = getattr(self._delegate, "get_provider", None) or getattr(
            self._delegate, "get", None
        )
        if getter is None:
            raise AttributeError("Provider registry does not expose get_provider() or get()")
        return getter(name)

    def list_providers(self) -> list[str]:
        return list(self._delegate.list_providers())


class _VectorStoreAdapter:
    """Adapter exposing legacy vector stores through the SDK protocol."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def index_document(
        self,
        doc_id: str,
        content: str,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        await self._delegate.add_documents(
            [content],
            [embedding],
            metadata=None if metadata is None else [metadata],
            ids=[doc_id],
        )

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
    ) -> list[Any]:
        return await self._delegate.search(query_embedding, top_k=limit)

    async def delete_document(self, doc_id: str) -> None:
        await self._delegate.delete([doc_id])
