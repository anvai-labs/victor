# Orchestrator Decomposition

> **Historical extraction record.** The topology and line counts below describe earlier decomposition phases. Use the [current runtime ownership](../architecture.md#service-layer) and [FEP-0031 target](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0031-chat-runtime-inversion.md#target-ownership-diagram) for present and proposed ownership. The [hotspot guard](https://github.com/anvai-labs/victor/blob/develop/tests/unit/runtime/test_hotspot_size_guard.py) tracks current size caps.

## Overview

The `AgentOrchestrator` (~4,593 LOC) is a facade coordinating extracted components, decomposed from a larger monolith through multiple refactoring phases. Ongoing work continues to shrink the facade.

## Component Topology

```
AgentOrchestrator (~4,593 LOC facade)
├── CallbackCoordinator (96 LOC)    ← tool/streaming lifecycle callbacks
│   ├── MetricsCoordinator          ← metrics collection + RL reward signals
│   ├── ToolCoordinator             ← tool completion handling + file tracking
│   └── UsageAnalytics              ← session analytics
├── OrchestratorPropertyFacade (496 LOC) ← 37 property definitions
│   ├── Simple accessors (17)       ← conversation_controller, tool_pipeline, etc.
│   ├── Lazy coordinators (8)       ← protocol_adapter, execution_coordinator, etc.
│   ├── Recovery properties (2)     ← recovery_handler, recovery_integration
│   └── Session state (10)          ← tool_calls_used, observed_files, etc. (with setters)
├── InitializationPhaseManager (153 LOC) ← 9-phase structured init
├── SessionStateAccessor            ← session state delegation to SessionStateManager
├── Runtime Boundaries (12 modules in victor/agent/runtime/)
│   ├── ProviderRuntime             ← provider coordinator + ProviderPool (feature-flagged)
│   ├── MetricsRuntime              ← usage logger, streaming metrics, cost tracker
│   ├── WorkflowRuntime             ← lazy workflow registry
│   ├── MemoryRuntime               ← memory manager + conversation embedding store
│   ├── ResilienceRuntime           ← recovery handler + integration
│   ├── CoordinationRuntime         ← recovery/chunk/planner/task coordinators
│   ├── InteractionRuntime          ← service-first chat/tool/session surfaces + compat shims
│   └── ServicesRuntime             ← DI service layer (Strangler Fig pattern)
├── 15 Coordinators (victor/agent/coordinators/)
│   ├── ExecutionCoordinator        ← agentic loop execution
│   ├── SyncChatCoordinator         ← deprecated non-streaming compatibility shim
│   ├── StreamingChatCoordinator    ← deprecated streaming compatibility shim
│   ├── UnifiedChatCoordinator      ← deprecated sync/streaming facade shim
│   ├── ProtocolAdapter             ← DIP compliance adapter
│   ├── MetricsCoordinator          ← centralized metrics
│   ├── SafetyCoordinator           ← safety rule evaluation
│   ├── ConversationCoordinator     ← conversation management
│   └── ... (7 more)
└── Vertical Integration
    ├── VerticalContext              ← unified vertical state container
    ├── VerticalIntegrationAdapter   ← single-source vertical method delegation
    └── ModeWorkflowTeamCoordinator ← intelligent team/workflow suggestions (lazy)
```

## Extraction Summary

| Component | LOC in Orchestrator | LOC Saved | Phase |
|-----------|-------------------|-----------|-------|
| OrchestratorPropertyFacade | 496 (in own file) | -574 from orchestrator | 3B |
| CallbackCoordinator | 96 (in own file) | -40 | 3A |
| SessionStateAccessor | — (prior work) | -100 | — |
| InitializationPhaseManager | 153 (in own file) | structural | 3C |

## Property Installation Pattern

Properties are installed as real class-level `property` descriptors via `install_properties()` at module load time. This preserves `unittest.mock.patch.object` compatibility:

```python
# orchestrator_properties.py
_PROPERTY_REGISTRY = {
    "conversation_controller": (_conversation_controller_get, None),
    "tool_calls_used": (_tool_calls_used_get, _tool_calls_used_set),
    # ... 37 total
}

def install_properties(cls):
    for name, (getter, setter) in _PROPERTY_REGISTRY.items():
        setattr(cls, name, property(getter, setter, doc=getter.__doc__))

# orchestrator.py (bottom of file)
install_properties(AgentOrchestrator)
```

All property access patterns are preserved:
```python
orchestrator.conversation_controller   # simple accessor
orchestrator.protocol_adapter          # lazy-inits on first access
orchestrator.tool_calls_used = 5       # setter works via property
```

Current chat/runtime note:
- Canonical chat entry points are `ChatService` and `ServiceStreamingRuntime`
- Deprecated chat coordinators remain only as compatibility import/runtime shims

## Extension Loader Decomposition

The `VerticalExtensionLoader` (1,897 LOC) delegates to three extracted components:

| Component | LOC | Purpose |
|-----------|-----|---------|
| ExtensionModuleResolver | 259 | Module path resolution, availability checking, class name generation |
| ExtensionCacheManager | 167 | Thread-safe namespaced caching with get_or_create/invalidate |
| ExtensionLoaderPressureMonitor | ~150 | Metrics counters, queue pressure thresholds, cooldown logic |
| CapabilityNegotiator | 111 | Validates ExtensionManifest during vertical activation |

### Extension Loading Flow

```
VerticalLoader.load(name)
  → VerticalRegistry.get(name) or entry point discovery
  → VerticalRuntimeAdapter.as_runtime_vertical_class()
  → _negotiate_manifest() → CapabilityNegotiator.negotiate(manifest)
  → _activate() → set as active vertical
```

### Module Resolution Flow

```
get_safety_extension() / get_middleware() / etc.
  → _find_available_candidates(suffix)
      → _module_resolver.resolve_candidates(vertical_name, suffix)
      → filter by _module_resolver.is_available(path)
  → _resolve_factory_extension(key, suffix)
      → _get_extension_factory(key, module_path)
          → _module_resolver.auto_generate_class_name()
          → _module_resolver.load_attribute()
      → _cache_manager.get_or_create(namespace, key, factory)
```

## ProviderPool

Single `ProviderPool` class in `victor/providers/factory.py` (duplicate removed). Wired into `ProviderRuntimeComponents` via `use_provider_pooling` feature flag in `FeatureFlagSettings`. Pool cleanup runs during `graceful_shutdown()`.
