# Contracts Boundary Architecture

This document describes the boundary between the contract package currently
published as `victor-contracts` and `victor-ai` (the framework runtime). New code
should use the semantic `victor_contracts` import alias and target the future
`victor-contracts` distribution name; `victor_contracts` remains a compatibility
namespace during the transition. External verticals depend only on the contract
package.

## Overview

```
External Vertical (victor-coding, etc.)
    │
    ▼
victor-contracts / victor-contracts compatibility
            ← Zero dependencies on victor-ai
    │           Provides: VerticalBase, PluginContext, VictorPlugin,
    │                     ExtensionManifest, ToolProvider, MockPluginContext
    ▼
victor-ai   ← Framework runtime
                Provides: AgentOrchestrator, ProviderRegistry, ToolExecutor,
                          CapabilityNegotiator, entry point loading
```

## Contract Package (`victor-contracts/`, future `victor-contracts`)

The contract package has **zero dependencies** on `victor-ai`. Its only runtime
dependency is `typing-extensions>=4.9`.

### Key Protocols

| Protocol | File | Purpose |
|----------|------|---------|
| `VerticalBase` | `victor_contracts/verticals/protocols/base.py` | Abstract base for all verticals |
| `PluginContext` | `victor_contracts/core/plugins.py` | DI interface for plugin registration |
| `VictorPlugin` | `victor_contracts/core/plugins.py` | Plugin lifecycle protocol |
| `ToolProvider` | `victor_contracts/verticals/protocols/tools.py` | Tool registration protocol |
| `ExtensionManifest` | `victor_contracts/verticals/manifest.py` | Capability declaration |

### Testing Utilities

| Utility | File | Purpose |
|---------|------|---------|
| `MockPluginContext` | `victor_contracts/testing/fixtures.py` | In-memory PluginContext for testing without victor-ai |
| `assert_valid_vertical()` | `victor_contracts/testing/__init__.py` | Validate vertical implements SDK contract |
| `assert_import_boundaries()` | `victor_contracts/testing/__init__.py` | Check vertical doesn't import victor internals |
| `validate_manifest()` | `victor_contracts/verticals/validation.py` | Static manifest completeness check |
| `audit_vertical_dependencies()` | `victor_contracts/verticals/validation.py` | Compare imports vs declared deps |

## Entry Points

Verticals register via `victor.plugins` entry point group in `pyproject.toml`:

```toml
[project.entry-points."victor.plugins"]
my_vertical = "my_package:plugin"
```

The framework discovers plugins via `victor/framework/entry_point_loader.py`
using `importlib.metadata` (NOT `sys.path` scanning). Additional entry point
groups:

- `victor.plugins` — VictorPlugin implementations
- `victor.extension.protocols` — preferred protocol-provider entry points
- `victor.extension.capabilities` — preferred capability-provider entry points
- `victor.sdk.protocols`, `victor.sdk.capabilities` — legacy compatibility
  groups supported during the contracts rename
- `victor.skills` — Skill definitions
- `victor.safety_rules`, `victor.tool_dependencies`, `victor.rl_configs`, `victor.escape_hatches`, `victor.commands`, `victor.prompt_contributors`, `victor.mode_configs`, `victor.workflow_providers`, `victor.team_spec_providers`, `victor.capability_providers`, `victor.service_providers`

## Manifest Validation Lifecycle

```
1. Definition    @register_vertical(name="my-vert", version="1.0.0")
                 → Attaches ExtensionManifest to class._victor_manifest

2. Discovery     VerticalRegistry.discover_external_verticals()
                 → Scans victor.plugins entry point group

3. Negotiation   CapabilityNegotiator.negotiate(manifest)
                 → Validates: API version, required extensions, extension_dependencies
                 → File: victor/core/verticals/capability_negotiator.py

4. Activation    VictorPlugin.register(context: PluginContext)
                 → Plugin registers tools, verticals, commands via context

5. Runtime       Orchestrator resolves tools via registered vertical providers
```

### Extension Dependency Validation

Manifests can declare dependencies on other extensions:

```python
ExtensionManifest(
    name="my-vert",
    extension_dependencies=[
        ExtensionDependency(extension_name="lancedb", min_version=">=0.4"),
        ExtensionDependency(extension_name="victor-coding", optional=True),
    ],
)
```

`CapabilityNegotiator._validate_extension_dependencies()` checks:
- Required deps are installed (hard error if missing)
- Version constraints are satisfied
- Optional deps produce warnings only

## Import Boundaries

Layer rule enforced by `scripts/check_imports.py`:

```
config/ ← providers/ ← tools/ ← agent/ ← ui/
```

Third-party vertical definition layers should import from `victor_contracts` only
(not `victor.core`, `victor.agent`, etc.). The five first-party domain verticals
are authored and published from the top-level `verticals/` packages in this
monorepo; they are not bundled under a second `victor/verticals/contrib/` source
tree. See [ADR-007](adr/007-vertical-distribution-and-sdk-boundary.md) for the
authoritative ownership and distribution boundary.

Contract boundary stability is enforced by contract shape tests at
`tests/unit/contracts/test_contracts_contract_shapes.py`.

## External Vertical Development

1. Depend on `victor-contracts>=X.Y` (not `victor-ai`)
2. Use `@register_vertical()` decorator or set `_victor_manifest`
3. Implement `VictorPlugin.register(context)` in your entry point
4. Declare `extension_dependencies` in manifest for third-party deps
5. Test with `MockPluginContext` from `victor_contracts.testing`
6. Run `validate_manifest(MyVertical)` and `audit_vertical_dependencies(src_dir, manifest)` in CI
