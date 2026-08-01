# Victor Contracts

Protocol and type definitions for building Victor verticals without pulling in
the Victor runtime.

## Overview

Use the contracts package when you want to author or publish a vertical package.
Use `victor-ai` when you want to run a vertical inside the Victor host runtime.

The supported external authoring model is contract-first:

- vertical packages depend on `victor-contracts`
- verticals declare tools, capabilities, prompts, teams, and workflow metadata through
  the contracts package
- `victor-ai` remains responsible for runtime concerns such as agent creation,
  capability injection, and tool execution

## Installation

### Contract-only authoring

```bash
pip install victor-contracts
```

### Runtime usage

```bash
pip install victor-ai
```

## Stable Contract Surface

The core authoring surface is available from the top-level package:

```python
from victor_contracts import (
    CURRENT_DEFINITION_VERSION,
    CapabilityIds,
    CapabilityRequirement,
    ToolNames,
    ToolRequirement,
    VerticalBase,
    VerticalDefinition,
)
```

Key pieces:

- `VerticalBase`: contract base class for external verticals
- `ToolNames`: canonical contract-owned tool identifiers
- `CapabilityIds`: canonical contract-owned runtime capability identifiers
- `ToolRequirement` / `CapabilityRequirement`: typed requirement declarations
- `VerticalDefinition`: validated serializable manifest returned by
  `get_definition()`

## Quick Start

```python
from victor_contracts import (
    CapabilityIds,
    CapabilityRequirement,
    ToolNames,
    ToolRequirement,
    VerticalBase,
)


class SecurityVertical(VerticalBase):
    name = "security"
    description = "Security analysis and audit workflows"
    version = "1.0.0"

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tool_requirements(cls) -> list[ToolRequirement]:
        return [
            ToolRequirement(ToolNames.READ, purpose="inspect code and configs"),
            ToolRequirement(ToolNames.GREP, purpose="search for vulnerable patterns"),
            ToolRequirement(ToolNames.SHELL, required=False, purpose="run scanners"),
            ToolRequirement(ToolNames.WEB_SEARCH, required=False, purpose="look up CVEs"),
        ]

    @classmethod
    def get_capability_requirements(cls) -> list[CapabilityRequirement]:
        return [
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="read repository contents",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.WEB_ACCESS,
                optional=True,
                purpose="fetch external security references",
            ),
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a security-focused assistant."

    @classmethod
    def get_prompt_templates(cls) -> dict[str, str]:
        return {
            "audit": "Audit the target for security issues and explain severity."
        }

    @classmethod
    def get_task_type_hints(cls) -> dict[str, dict[str, object]]:
        return {
            "audit": {
                "hint": "Start with read-first reconnaissance, then validate findings.",
                "tool_budget": 12,
                "priority_tools": [ToolNames.READ, ToolNames.GREP, ToolNames.SHELL],
            }
        }

    @classmethod
    def get_team_declarations(cls) -> dict[str, dict[str, object]]:
        return {
            "security_review_team": {
                "name": "Security Review Team",
                "formation": "pipeline",
                "members": [
                    {
                        "role": "researcher",
                        "goal": "Inspect the target and identify likely risks.",
                    },
                    {
                        "role": "reviewer",
                        "goal": "Validate findings before escalation.",
                    },
                ],
            }
        }


definition = SecurityVertical.get_definition()
assert definition.definition_version == "1.0"
assert definition.tools == [ToolNames.READ, ToolNames.GREP, ToolNames.SHELL, ToolNames.WEB_SEARCH]
assert definition.team_metadata.teams[0].team_id == "security_review_team"
```

## Definition Contract

`VerticalBase.get_definition()` is the preferred contract API. It returns a
validated `VerticalDefinition` manifest that contains:

- `definition_version`
- canonical tool identifiers
- typed tool and capability requirements
- system prompt text
- prompt metadata and task-type hints
- team metadata such as declarative team layouts and the default team identifier
- declarative stage definitions
- workflow metadata such as initial stage, provider hints, and evaluation criteria

Compatibility note:

- `get_config()` still exists as a bridge for current victor-ai integrations
- `VerticalDefinition.to_config()` / `from_config()` bridge the legacy config shape
- `VerticalDefinition.from_dict()` supports serialized manifest round-tripping

## Packaging

Register the vertical via the standard Victor entry point:

```toml
[project]
name = "victor-security"
version = "1.0.0"
dependencies = ["victor-contracts>=1.0.0"]

[project.entry-points."victor.plugins"]
security = "victor_security:plugin"
```

## Discovery

Victor discovers external vertical packages through `victor.plugins` entry
points, then each plugin registers one or more contract vertical definitions:

```python
from victor_contracts.discovery import get_global_registry

registry = get_global_registry()
verticals = registry.get_verticals()
```

## Guides And Examples

- [Vertical Development Guide](VERTICAL_DEVELOPMENT.md)
- [Contract Stability Policy](CONTRACT_STABILITY.md)
- [Changelog](CHANGELOG.md)
- [Migration Guide](MIGRATION.md)
- [Minimal contract-only example](examples/minimal_vertical/README.md)
- [Repository external package example](../examples/external_vertical/README.md)

## Testing

```bash
pip install -e ".[dev]"
pytest victor-contracts/tests -q
```

To verify compatibility with the runtime as well:

```bash
pip install -e ".[dev]" victor-ai
pytest victor-contracts/tests/unit tests/integration/test_contracts_integration.py -q
```

## Versioning

The package follows semantic versioning, and the manifest contract is versioned
separately via `VerticalDefinition.definition_version`. External verticals
should treat the contracts package as the source of truth for supported identifiers
and manifest fields.

Per-surface stability tiers and the deprecation policy (warn at least one minor
release before removal; removals only in minor bumps) are defined in
[CONTRACT_STABILITY.md](CONTRACT_STABILITY.md). Releases are recorded in
[CHANGELOG.md](CHANGELOG.md).

## Links

- Repository: https://github.com/anvai-labs/victor
- Issues: https://github.com/anvai-labs/victor/issues
