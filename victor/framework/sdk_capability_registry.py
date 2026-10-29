"""Runtime registry for resolving SDK-declared capability requirements."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from victor.framework.capability_runtime import check_capability
from victor_contracts.constants import CapabilityIds
from victor_contracts.core.types import (
    CapabilityRequirementLike,
    normalize_capability_requirements,
)


def _load_object(import_path: str) -> Any:
    """Load an object from ``module:attribute`` import path syntax."""

    module_name, attribute_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


@dataclass(frozen=True)
class RuntimeCapabilityBinding:
    """Mapping between an SDK capability ID and runtime satisfaction checks."""

    capability_id: str
    description: str
    orchestrator_capabilities: Tuple[str, ...] = ()
    required_tools: Tuple[str, ...] = ()
    any_tools: Tuple[str, ...] = ()
    provider_protocol_imports: Tuple[str, ...] = ()
    builtin_provider_imports: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityResolution:
    """Resolution result for one SDK capability requirement."""

    capability_id: str
    available: bool
    optional: bool
    known: bool
    source: Optional[str] = None
    reason: Optional[str] = None
    min_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of this resolution."""

        payload: Dict[str, Any] = {
            "capability_id": self.capability_id,
            "available": self.available,
            "optional": self.optional,
            "known": self.known,
            "metadata": dict(self.metadata),
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.min_version is not None:
            payload["min_version"] = self.min_version
        return payload


class SdkCapabilityNegotiator:
    """Registry of runtime bindings for SDK capability identifiers.

    Negotiates SDK capability requirements against runtime satisfactions.
    Maps SDK capability IDs (e.g., CapabilityIds.CODEBASE_INDEXING) to
    orchestrator capabilities, required tools, and provider protocols.

    Note: Previously named RuntimeCapabilityRegistry - renamed for clarity.
    """

    def __init__(self) -> None:
        self._bindings: Dict[str, RuntimeCapabilityBinding] = {}

    def register(self, binding: RuntimeCapabilityBinding) -> None:
        """Register a binding for a capability identifier."""

        self._bindings[binding.capability_id] = binding

    def get(self, capability_id: str) -> Optional[RuntimeCapabilityBinding]:
        """Return the registered binding for a capability ID, if any."""

        return self._bindings.get(capability_id)

    def list_bindings(self) -> List[RuntimeCapabilityBinding]:
        """Return all bindings in registration order."""

        return list(self._bindings.values())

    def resolve_requirement(
        self,
        requirement: CapabilityRequirementLike,
        *,
        orchestrator: Any = None,
        available_tools: Optional[Iterable[str]] = None,
    ) -> CapabilityResolution:
        """Resolve one capability requirement against the current runtime."""

        normalized = normalize_capability_requirements([requirement])[0]
        binding = self.get(normalized.capability_id)
        if binding is None:
            return CapabilityResolution(
                capability_id=normalized.capability_id,
                available=False,
                optional=normalized.optional,
                known=False,
                reason="No runtime binding is registered for this SDK capability ID.",
                min_version=normalized.min_version,
            )

        tool_names = {str(tool_name) for tool_name in (available_tools or [])}

        if orchestrator is not None:
            for capability_name in binding.orchestrator_capabilities:
                if check_capability(
                    orchestrator,
                    capability_name,
                    min_version=normalized.min_version,
                ):
                    return CapabilityResolution(
                        capability_id=normalized.capability_id,
                        available=True,
                        optional=normalized.optional,
                        known=True,
                        source=f"orchestrator:{capability_name}",
                        min_version=normalized.min_version,
                        metadata={"binding_type": "orchestrator_capability"},
                    )

        if binding.required_tools and set(binding.required_tools).issubset(tool_names):
            return CapabilityResolution(
                capability_id=normalized.capability_id,
                available=True,
                optional=normalized.optional,
                known=True,
                source=f"tools:{','.join(binding.required_tools)}",
                min_version=normalized.min_version,
                metadata={"binding_type": "tool_bundle"},
            )

        if binding.any_tools:
            for tool_name in binding.any_tools:
                if tool_name in tool_names:
                    return CapabilityResolution(
                        capability_id=normalized.capability_id,
                        available=True,
                        optional=normalized.optional,
                        known=True,
                        source=f"tools:{tool_name}",
                        min_version=normalized.min_version,
                        metadata={"binding_type": "tool_bundle"},
                    )

        if binding.provider_protocol_imports:
            from victor.core.capability_registry import CapabilityRegistry

            registry = CapabilityRegistry.get_instance()
            for protocol_import in binding.provider_protocol_imports:
                protocol = _load_object(protocol_import)
                provider = registry.get(protocol)
                if provider is not None:
                    return CapabilityResolution(
                        capability_id=normalized.capability_id,
                        available=True,
                        optional=normalized.optional,
                        known=True,
                        source=f"provider:{protocol_import}",
                        min_version=normalized.min_version,
                        metadata={"binding_type": "core_capability_registry"},
                    )

        for provider_import in binding.builtin_provider_imports:
            provider_type = _load_object(provider_import)
            if provider_type is not None:
                return CapabilityResolution(
                    capability_id=normalized.capability_id,
                    available=True,
                    optional=normalized.optional,
                    known=True,
                    source=f"builtin:{provider_import}",
                    min_version=normalized.min_version,
                    metadata={"binding_type": "framework_builtin"},
                )

        reasons: List[str] = []
        if binding.orchestrator_capabilities:
            joined = ", ".join(binding.orchestrator_capabilities)
            reasons.append(f"missing orchestrator capability [{joined}]")
        if binding.required_tools:
            joined = ", ".join(binding.required_tools)
            reasons.append(f"missing required tools [{joined}]")
        if binding.any_tools:
            joined = ", ".join(binding.any_tools)
            reasons.append(f"missing any supported tool [{joined}]")
        if binding.provider_protocol_imports:
            reasons.append("no provider is registered in the core capability registry")
        if binding.builtin_provider_imports:
            reasons.append("framework provider import failed")

        return CapabilityResolution(
            capability_id=normalized.capability_id,
            available=False,
            optional=normalized.optional,
            known=True,
            reason="; ".join(reasons) or "Capability requirement could not be satisfied.",
            min_version=normalized.min_version,
            metadata={"binding_type": "unresolved"},
        )

    def resolve_requirements(
        self,
        requirements: Iterable[CapabilityRequirementLike],
        *,
        orchestrator: Any = None,
        available_tools: Optional[Iterable[str]] = None,
    ) -> List[CapabilityResolution]:
        """Resolve a sequence of requirements against the current runtime."""

        return [
            self.resolve_requirement(
                requirement,
                orchestrator=orchestrator,
                available_tools=available_tools,
            )
            for requirement in requirements
        ]


_RUNTIME_CAPABILITY_REGISTRY: Optional[RuntimeCapabilityRegistry] = None


def get_runtime_capability_registry(*, reset: bool = False) -> RuntimeCapabilityRegistry:
    """Return the shared runtime capability registry."""

    global _RUNTIME_CAPABILITY_REGISTRY
    if reset or _RUNTIME_CAPABILITY_REGISTRY is None:
        registry = RuntimeCapabilityRegistry()
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.FILE_OPS,
                description="File operations are satisfied by the core filesystem tool bundle.",
                required_tools=("read", "write", "edit", "grep"),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.SHELL_ACCESS,
                description="Shell access is satisfied by the shell tool.",
                any_tools=("shell",),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.GIT,
                description="Git access is satisfied by the git tool.",
                any_tools=("git",),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.CONTAINER_RUNTIME,
                description="Container runtime access is satisfied by Docker-style tooling.",
                any_tools=("docker",),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.LSP,
                description="Language intelligence requires an LSP runtime capability or provider.",
                orchestrator_capabilities=("lsp",),
                provider_protocol_imports=(
                    "victor.framework.vertical_protocols:LSPManagerProtocol",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.WEB_ACCESS,
                description="Web access is satisfied by any enabled network retrieval tool.",
                any_tools=("web_search", "web_fetch", "http_request", "fetch_url"),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.DOCUMENT_INGESTION,
                description="Document ingestion is satisfied by the RAG ingestion tool bundle.",
                required_tools=("rag_ingest", "read", "ls"),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.RETRIEVAL,
                description="Retrieval is satisfied by the RAG query/search tool bundle.",
                required_tools=("rag_query", "rag_search"),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.VECTOR_INDEXING,
                description="Vector indexing is satisfied by RAG ingestion or query tooling.",
                any_tools=("rag_ingest", "rag_query"),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.PROMPT_CONTRIBUTIONS,
                description="Prompt contribution support is provided by framework prompt capabilities.",
                orchestrator_capabilities=(
                    "task_type_hints",
                    "prompt_section",
                    "prompt_builder",
                ),
                builtin_provider_imports=(
                    "victor.framework.capabilities.prompt_contributions:PromptContributionCapability",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.PRIVACY,
                description="Privacy controls are provided by the framework privacy capability provider.",
                builtin_provider_imports=(
                    "victor.framework.capabilities.privacy:PrivacyCapabilityProvider",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.SECRETS_MASKING,
                description="Secrets masking is provided by the framework privacy/middleware layer.",
                builtin_provider_imports=(
                    "victor.framework.capabilities.privacy:PrivacyCapabilityProvider",
                    "victor.framework.middleware:SecretMaskingMiddleware",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.AUDIT_LOGGING,
                description="Audit logging is provided by the framework privacy/audit layer.",
                builtin_provider_imports=(
                    "victor.framework.capabilities.privacy:PrivacyCapabilityProvider",
                    "victor.security.audit.manager:AuditManager",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.STAGES,
                description="Stage definitions are provided by the framework stage builder capability.",
                builtin_provider_imports=(
                    "victor.framework.capabilities.stages:StageBuilderCapability",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.GROUNDING_RULES,
                description="Grounding rules are provided by the framework grounding rules capability.",
                builtin_provider_imports=(
                    "victor.framework.capabilities.grounding_rules:GroundingRulesCapability",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.VALIDATION,
                description="Validation is provided by the framework validation capability provider.",
                builtin_provider_imports=(
                    "victor.framework.capabilities.validation:ValidationCapabilityProvider",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.SAFETY_RULES,
                description="Safety rules are provided by the framework safety rules capability provider.",
                orchestrator_capabilities=(
                    "vertical_safety_patterns",
                    "safety_patterns",
                ),
                builtin_provider_imports=(
                    "victor.framework.capabilities.safety_rules:SafetyRulesCapabilityProvider",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.TASK_HINTS,
                description="Task hints are provided by the framework task-hint capability provider.",
                orchestrator_capabilities=("task_type_hints",),
                builtin_provider_imports=(
                    "victor.framework.capabilities.task_hints:TaskTypeHintCapabilityProvider",
                ),
            )
        )
        registry.register(
            RuntimeCapabilityBinding(
                capability_id=CapabilityIds.SOURCE_VERIFICATION,
                description="Source verification is provided by the framework source-verification capability provider.",
                builtin_provider_imports=(
                    "victor.framework.capabilities.source_verification:SourceVerificationCapabilityProvider",
                ),
            )
        )
        _RUNTIME_CAPABILITY_REGISTRY = registry

    return _RUNTIME_CAPABILITY_REGISTRY


def resolve_capability_requirement(
    requirement: CapabilityRequirementLike,
    *,
    orchestrator: Any = None,
    available_tools: Optional[Iterable[str]] = None,
    registry: Optional[RuntimeCapabilityRegistry] = None,
) -> CapabilityResolution:
    """Resolve one SDK capability requirement against the runtime."""

    active_registry = registry or get_runtime_capability_registry()
    return active_registry.resolve_requirement(
        requirement,
        orchestrator=orchestrator,
        available_tools=available_tools,
    )


def resolve_capability_requirements(
    requirements: Iterable[CapabilityRequirementLike],
    *,
    orchestrator: Any = None,
    available_tools: Optional[Iterable[str]] = None,
    registry: Optional["SdkCapabilityNegotiator"] = None,
) -> List[CapabilityResolution]:
    """Resolve SDK capability requirements against the runtime."""

    active_registry = registry or get_runtime_capability_registry()
    return active_registry.resolve_requirements(
        requirements,
        orchestrator=orchestrator,
        available_tools=available_tools,
    )


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# Alias for backward compatibility - was renamed for clarity
RuntimeCapabilityRegistry = SdkCapabilityNegotiator


__all__ = [
    "CapabilityResolution",
    "RuntimeCapabilityBinding",
    "SdkCapabilityNegotiator",
    "RuntimeCapabilityRegistry",  # Backward compat alias
    "get_runtime_capability_registry",
    "resolve_capability_requirement",
    "resolve_capability_requirements",
]
