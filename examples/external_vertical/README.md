# Victor Security Vertical - Contract-Only External Example

This example shows the supported external package model for Victor verticals:
author the vertical against the `victor_contracts` import surface from
`victor-contracts`, publish it as a normal Python package, and let `victor-ai`
discover it at runtime through the `victor.plugins` entry point.

## What This Example Demonstrates

- Contract-only package dependency for authoring
- canonical `ToolNames` and `CapabilityIds`
- manifest-first vertical definition via `get_definition()`
- entry-point based runtime discovery
- a custom duck-typed tool registered via `PluginContext.register_tool()`
- a mode-config provider discovered via the `victor.mode_configs` entry point
- contract-only tests that run without `victor-ai` installed

## Package Structure

```text
examples/external_vertical/
├── pyproject.toml
├── README.md
├── victor-vertical.toml        # mirror of src/victor_security/victor-vertical.toml
├── src/
│   └── victor_security/
│       ├── __init__.py         # SecurityPlugin (victor.plugins entry point)
│       ├── assistant.py        # SecurityAssistant vertical definition
│       ├── mode_config.py      # SecurityModeConfigProvider
│       ├── tools.py            # SecretPatternScanTool custom tool
│       └── victor-vertical.toml
└── tests/
    ├── test_definition.py
    ├── test_mode_config.py
    └── test_plugin.py
```

The canonical package metadata file is
`src/victor_security/victor-vertical.toml` (shipped inside the wheel); the
top-level `victor-vertical.toml` is a byte-identical mirror kept for
repository browsing.

## Installation

### Contract-only authoring

```bash
cd examples/external_vertical
pip install -e .
```

This installs the package with only `victor-contracts` as the distribution dependency
and imports public contracts through `victor_contracts`.

### With Victor runtime

```bash
cd examples/external_vertical
pip install -e ".[runtime]"
```

This also installs `victor-ai`, which can discover and run the vertical.

## Authoring Example

```python
from victor_security import SecurityAssistant

definition = SecurityAssistant.get_definition()

print(definition.name)
print(definition.tools)
print(definition.capability_requirements)
print(definition.workflow_metadata.workflow_spec)
```

The preferred contract is `get_definition()`. `get_config()` remains available
as a compatibility bridge for current runtime integrations.

## Custom Tools

`src/victor_security/tools.py` ships `SecretPatternScanTool`, a
defensive-security tool that scans text and files for hardcoded secret
patterns (cloud keys, API tokens, private key blocks) and reports masked
matches. It is duck-typed against the tool shape the Victor runtime accepts
from plugins - no base class import is required:

- `name` - stable tool identifier (`secret_pattern_scan`)
- `description` - what the tool does, for LLM tool selection
- `parameters` - JSON-schema dict describing the arguments
- `async execute(**kwargs)` - the tool implementation

The plugin registers it inside `SecurityPlugin.register()`:

```python
def register(self, context: PluginContext) -> None:
    context.register_vertical(SecurityAssistant)
    context.register_tool(SecretPatternScanTool())
```

When the package is installed next to `victor-ai`, the host's
`PluginContext` routes the instance into the framework `ToolRegistry`, so
agents can call `secret_pattern_scan` like any built-in tool.

## Mode Configs

`src/victor_security/mode_config.py` defines `SecurityModeConfigProvider`
using the SDK-owned `StaticModeConfigProvider` and `ModeDefinition` types
from `victor_contracts.verticals.mode_config`. It declares `quick` and
`deep` modes whose `allowed_stages` match the assistant's stages
(`reconnaissance`, `analysis`, `reporting`) and task budgets that mirror
`get_task_type_hints()`.

The runtime discovers the provider through the `victor.mode_configs` entry
point declared in `pyproject.toml`:

```toml
[project.entry-points."victor.mode_configs"]
security = "victor_security.mode_config:SecurityModeConfigProvider"
```

## Running The Example's Tests

The tests import only `victor_security` and `victor_contracts`, so they run
in a contract-only environment (no `victor-ai` needed):

```bash
cd examples/external_vertical
pip install -e ".[test]"
pytest
```

## Validating The Package Contract

`victor-contracts` ships a CLI that validates an installed vertical package
(entry points, definition surface, `victor-vertical.toml` metadata):

```bash
victor-contracts check victor-security
```

## Runtime Usage

After installing the runtime extra, Victor can discover the package through the
entry point declared in `pyproject.toml`:

```toml
[project.entry-points."victor.plugins"]
security = "victor_security:plugin"
```

Examples:

```bash
victor --vertical security
victor --list-verticals
```

To load the vertical programmatically under `victor-ai` (the same path the
CLI uses), go through the vertical loader:

```python
from victor.core.verticals.vertical_loader import VerticalLoader

loader = VerticalLoader()
discovered = loader.discover_verticals(force_refresh=True)
assert "security" in discovered

vertical = loader.load("security")
definition = vertical.get_definition()
mode_provider = vertical.get_mode_config_provider()

print(definition.name)                      # "security"
print(sorted(mode_provider.get_mode_configs()))  # ["deep", "quick"]
```

## Key Contract Choices In This Example

- `victor_contracts.VerticalBase` is the only base class used by the package
- tools are declared with `ToolRequirement` and `ToolNames`
- runtime needs are declared with `CapabilityRequirement` and `CapabilityIds`
- prompt templates, task hints, stages, team layouts, and workflow metadata are all expressed
  through SDK hooks
- no `victor.core` or `victor.framework` imports are required to author the package

## Next Steps

- Compare this example with [victor-contracts/README.md](../../victor-contracts/README.md)
- See the broader SDK guide in [victor-contracts/VERTICAL_DEVELOPMENT.md](../../victor-contracts/VERTICAL_DEVELOPMENT.md)
