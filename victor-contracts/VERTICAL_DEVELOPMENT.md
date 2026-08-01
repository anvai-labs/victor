# Vertical Development Guide

Complete guide for developing Victor verticals from scratch.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Creating a Basic Vertical](#creating-a-basic-vertical)
4. [Advanced Features](#advanced-features)
5. [Testing Your Vertical](#testing-your-vertical)
6. [Publishing Your Vertical](#publishing-your-vertical)
7. [Best Practices](#best-practices)
8. [Definition Contract](#definition-contract)

---

## Overview

A **vertical** is a domain-specific assistant built on the Victor framework. Verticals provide:

- **Domain Expertise**: Specialized system prompts and knowledge
- **Tool Configuration**: Curated tool sets for specific tasks
- **Workflow Definitions**: Multi-stage processes tailored to the domain
- **Safety Rules**: Domain-specific safety constraints
- **Custom Capabilities**: Unique features for the domain

### Supported External Authoring Model

For external packages, the supported contract is contract-first:

- depend on `victor-contracts` only
- import `VerticalBase`, `ToolNames`, and `CapabilityIds` from `victor_contracts`
- declare requirements and metadata through `get_definition()` or the class hooks
  that feed it
- leave runtime concerns such as agent creation and capability injection to
  `victor-ai`

Which import surfaces are stable, which require an installed `victor-ai` host,
and how deprecations are handled is defined in
[CONTRACT_STABILITY.md](CONTRACT_STABILITY.md).

### Example Verticals

- **victor-coding**: Software development assistant
- **victor-research**: Research and writing assistant
- **victor-devops**: DevOps and infrastructure assistant
- **victor-rag**: Retrieval-augmented generation assistant

---

## Project Structure

### Minimal Structure

```
my-vertical/
├── pyproject.toml
├── my_vertical/
│   ├── __init__.py
└── README.md
```

### Full Structure (Advanced)

```
my-vertical/
├── pyproject.toml
├── README.md
├── my_vertical/
│   ├── __init__.py           # Vertical definition
│   ├── protocols.py          # Protocol implementations
│   ├── capabilities.py       # Capability providers
│   ├── validators.py         # Validation functions
│   └── workflows/
│       ├── __init__.py
│       └── custom.py         # Custom workflow definitions
└── tests/
    ├── __init__.py
    └── test_vertical.py      # Unit tests
```

---

## Creating a Basic Vertical

### Step 1: Create pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "victor-my-vertical"
version = "1.0.0"
description = "My custom Victor vertical"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [
    { name = "Your Name", email = "your.email@example.com" },
]

# ONLY depend on victor-contracts for zero runtime dependencies!
dependencies = [
    "victor-contracts>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/victor-my-vertical"
Repository = "https://github.com/yourusername/victor-my-vertical"
Documentation = "https://docs.victor.dev/verticals/my-vertical"

[tool.setuptools.packages.find]
where = ["."]
include = ["my_vertical*"]

# Register your vertical
[project.entry-points."victor.plugins"]
my-vertical = "my_vertical:plugin"
```

### Step 2: Define Your Vertical

```python
# my_vertical/__init__.py

from victor_contracts import (
    CapabilityIds,
    CapabilityRequirement,
    ToolNames,
    ToolRequirement,
    VerticalBase,
)


class MyVertical(VerticalBase):
    """A custom vertical for my specific use case.

    This vertical provides specialized assistance for X.
    """

    # Vertical metadata
    name = "my-vertical"
    description = "My custom vertical for X"
    version = "1.0.0"
    author = "Your Name"

    @classmethod
    def get_name(cls) -> str:
        """Return vertical identifier."""
        return "my-vertical"

    @classmethod
    def get_description(cls) -> str:
        """Return human-readable description."""
        return "Custom vertical for X, Y, and Z tasks"

    @classmethod
    def get_tool_requirements(cls) -> list[ToolRequirement]:
        """Return typed tool requirements for this vertical."""
        return [
            ToolRequirement(ToolNames.READ, purpose="inspect local files"),
            ToolRequirement(ToolNames.WRITE, purpose="modify reports or configs"),
            ToolRequirement(ToolNames.CODE_SEARCH, purpose="find relevant code paths"),
            ToolRequirement(ToolNames.SYMBOL, required=False, purpose="jump to symbols"),
            ToolRequirement(ToolNames.GIT, required=False, purpose="inspect repository history"),
            ToolRequirement(ToolNames.SHELL, required=False, purpose="run local checks"),
            ToolRequirement(ToolNames.TEST, required=False, purpose="validate changes"),
        ]

    @classmethod
    def get_capability_requirements(cls) -> list[CapabilityRequirement]:
        """Declare host/runtime capabilities needed by this vertical."""
        return [
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="read and write workspace files",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.GIT,
                optional=True,
                purpose="use repository-aware workflows when available",
            ),
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return system prompt for this vertical.

        The system prompt defines the agent's behavior and expertise.
        """
        return """You are an expert assistant for X with deep knowledge in Y and Z.

## Your Expertise

You specialize in:
- X task planning and execution
- Y best practices and patterns
- Z troubleshooting and optimization

## Your Approach

1. **Understand**: First understand the user's goal and context
2. **Plan**: Break down complex tasks into clear steps
3. **Execute**: Use available tools efficiently
4. **Verify**: Always verify your results
5. **Document**: Provide clear explanations

## Safety Guidelines

- Always verify file paths before operations
- Use version control (git) for changes
- Test changes before suggesting them
- Ask for clarification when uncertain

## Communication Style

- Be concise but thorough
- Provide examples when helpful
- Explain trade-offs and alternatives
- Flag potential issues early"""

    @classmethod
    def get_prompt_templates(cls) -> dict[str, str]:
        return {
            "analysis": "Analyze the target before making changes.",
            "implementation": "Implement the requested change and verify it.",
        }

    @classmethod
    def get_task_type_hints(cls) -> dict[str, dict[str, object]]:
        return {
            "analysis": {
                "hint": "Prefer read-first workflows and summarize findings clearly.",
                "tool_budget": 8,
                "priority_tools": [ToolNames.READ, ToolNames.CODE_SEARCH],
            }
        }
```

The preferred artifact for external verticals is the validated manifest returned
by `get_definition()`:

```python
definition = MyVertical.get_definition()
config = MyVertical.get_config()  # compatibility bridge for current runtime users
```

## Definition Contract

`VerticalDefinition` is the SDK-owned, serializable contract for external
verticals. It includes:

- `definition_version` for schema compatibility checks
- canonical tool IDs via `ToolNames`
- typed tool and capability requirements
- prompt metadata and task-type hints
- team metadata for declarative multi-agent layouts
- declarative stages and workflow metadata

Use the SDK-owned identifiers rather than runtime-owned string registries:

```python
from victor_contracts import CapabilityIds, ToolNames

tools = [ToolNames.READ, ToolNames.GREP, ToolNames.SHELL]
capabilities = [CapabilityIds.FILE_OPS, CapabilityIds.GIT]
```

### Step 3: Create README.md

```markdown
# victor-my-vertical

A custom Victor vertical for X.

## Features

- Expert assistance for X tasks
- Specialized tool configuration
- Custom workflow stages
- Built-in safety rules

## Installation

```bash
pip install victor-my-vertical
```

## Usage

```python
from victor import Agent

# Create agent with your vertical
agent = Agent(vertical="my-vertical")

# Use the agent
result = agent.run("Help me with X task")
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

## License

Apache-2.0
```

---

## Advanced Features

### Custom Stages

Define domain-specific workflow stages:

```python
from victor_contracts.core.types import StageDefinition

class MyVertical(VerticalBase):
    @classmethod
    def get_stages(cls):
        return {
            "analyze": StageDefinition(
                name="analyze",
                description="Analyze the problem and understand requirements",
                required_tools=["read", "search"],
                optional_tools=["code_search"],
            ),
            "design": StageDefinition(
                name="design",
                description="Design a solution approach",
                required_tools=["read"],
                optional_tools=["write"],
            ),
            "implement": StageDefinition(
                name="implement",
                description="Implement the solution",
                required_tools=["read", "write"],
                optional_tools=["shell", "test"],
            ),
            "validate": StageDefinition(
                name="validate",
                description="Validate and test the solution",
                required_tools=["test", "read"],
                optional_tools=["shell"],
            ),
        }
```

### Protocol Implementations

Add specialized protocols:

```python
# my_vertical/protocols.py

from typing import Dict, Any, List
from victor_contracts.verticals.protocols import ToolProvider, SafetyProvider


class MyToolProvider(ToolProvider):
    """Custom tool provider for my vertical."""

    def get_tools(self) -> List[str]:
        return ["read", "write", "search", "git"]

    def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]:
        """Optimize tools for specific stages."""
        if stage == "analyze":
            return ["read", "search", "code_search"]
        elif stage == "implement":
            return ["read", "write", "git"]
        elif stage == "validate":
            return ["read", "test"]
        return self.get_tools()


class MySafetyProvider(SafetyProvider):
    """Custom safety rules for my vertical."""

    def __init__(self):
        self._rules = {
            "protected_paths": [
                "/etc",
                "/sys",
                "/proc",
                "~/.ssh",
                "~/.aws",
            ],
            "max_file_size": 10 * 1024 * 1024,  # 10MB
        }

    def get_safety_rules(self) -> Dict[str, Any]:
        return self._rules.copy()

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Validate tool calls before execution."""
        if tool_name == "write":
            path = arguments.get("path", "")
            for protected in self._rules["protected_paths"]:
                if path.startswith(protected):
                    return False

        if tool_name == "shell":
            command = arguments.get("command", "")
            dangerous = ["rm -rf", "dd if=", "mkfs"]
            return not any(d in command for d in dangerous)

        return True

    def validate_prompt(self, prompt: str) -> bool:
        """Validate user prompts."""
        # Check for obviously malicious patterns
        malicious = [
            "delete all files",
            "format hard drive",
            "remove system files",
        ]
        return not any(m in prompt.lower() for m in malicious)
```

Register protocols in pyproject.toml:

```toml
[project.entry-points."victor.extension.protocols"]
my-tools = "my_vertical.protocols:MyToolProvider"
my-safety = "my_vertical.protocols:MySafetyProvider"
```

### Custom Capabilities

Add unique capabilities:

```python
# my_vertical/capabilities.py

from typing import Dict, Any, List


class MyAnalysisCapability:
    """Custom analysis capability for my vertical."""

    def __init__(self):
        self._patterns = {}
        self._metrics = {}

    def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code and return insights."""
        results = {
            "language": language,
            "lines": len(code.split("\n")),
            "complexity": self._calculate_complexity(code),
            "patterns": self._find_patterns(code),
        }
        return results

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        # Simplified complexity calculation
        count = 1
        for line in code.split("\n"):
            if any(kw in line for kw in ["if", "for", "while", "case", "catch"]):
                count += 1
        return count

    def _find_patterns(self, code: str) -> List[str]:
        """Find code patterns."""
        patterns = []
        if "class " in code:
            patterns.append("object-oriented")
        if "async def " in code:
            patterns.append("async")
        if "def " in code and "class " not in code:
            patterns.append("procedural")
        return patterns
```

Register capabilities:

```toml
[project.entry-points."victor.extension.capabilities"]
my-analysis = "my_vertical.capabilities:MyAnalysisCapability"
```

### Custom Validators

Add validation functions:

```python
# my_vertical/validators.py

from typing import Dict, Any


def validate_code_style(code: str, language: str = "python") -> Dict[str, Any]:
    """Validate code style."""
    issues = []

    if language == "python":
        # Check line length
        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > 100:
                issues.append(f"Line {i}: exceeds 100 characters")

        # Check for missing docstrings
        if "def " in code and '"""' not in code:
            issues.append("Missing module docstring")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


def validate_git_status(repo_path: str) -> Dict[str, Any]:
    """Validate git repository status."""
    import os

    if not os.path.exists(os.path.join(repo_path, ".git")):
        return {
            "valid": False,
            "issues": ["Not a git repository"],
        }

    return {
        "valid": True,
        "issues": [],
    }
```

Register validators:

```toml
[project.entry-points."victor.extension.validators"]
code-style = "my_vertical.validators:validate_code_style"
git-status = "my_vertical.validators:validate_git_status"
```

---

## Testing Your Vertical

### Unit Tests

```python
# tests/test_vertical.py

import pytest
from my_vertical import MyVertical


class TestMyVertical:
    def test_get_name(self):
        assert MyVertical.get_name() == "my-vertical"

    def test_get_description(self):
        assert "custom vertical" in MyVertical.get_description().lower()

    def test_get_tools(self):
        tools = MyVertical.get_tools()
        assert isinstance(tools, list)
        assert "read" in tools
        assert "write" in tools

    def test_get_system_prompt(self):
        prompt = MyVertical.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_get_config(self):
        config = MyVertical.get_config()
        assert config.name == "my-vertical"
        assert len(config.get_tool_names()) > 0

    def test_get_stages(self):
        stages = MyVertical.get_stages()
        assert isinstance(stages, dict)
        assert len(stages) > 0
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
from victor import Agent


@pytest.mark.integration
class TestMyVerticalIntegration:
    def test_create_agent(self):
        agent = Agent(vertical="my-vertical")
        assert agent is not None

    def test_simple_query(self):
        agent = Agent(vertical="my-vertical")
        result = agent.run("What tools do you have?")
        assert result is not None
```

---

## Publishing Your Vertical

### 1. Version Your Vertical

Use semantic versioning:

```python
# __init__.py
__version__ = "1.0.0"
```

Update in pyproject.toml:

```toml
[project]
version = "1.0.0"
```

### 2. Build Your Package

```bash
pip install build
python -m build
```

### 3. Publish to PyPI

```bash
pip install twine
twine upload dist/*
```

### 4. Announce Your Vertical

- Add to Victor vertical registry
- Publish documentation
- Share with the community

---

## Best Practices

### 1. Start Simple

Begin with a basic vertical, then add advanced features:

```python
# Start here
class MyVertical(VerticalBase):
    @classmethod
    def get_tools(cls):
        return ["read", "write"]

    @classmethod
    def get_system_prompt(cls):
        return "You are helpful."

# Then add features incrementally
```

### 2. Use Descriptive Names

Choose clear, descriptive names:

```python
# GOOD
name = "data-analysis"
description = "Data analysis and visualization assistant"

# AVOID
name = "my-vertical"
description = "Does stuff"
```

### 3. Document Your Tools

Explain why each tool is needed:

```python
def get_tools(cls):
    return [
        "read",      # For reading data files
        "write",     # For writing reports
        "search",    # For finding relevant data
        "database",  # For querying databases
    ]
```

For new external verticals, prefer `get_tool_requirements()` with
`ToolRequirement` so each tool has an explicit purpose and optionality.

### 4. Provide Good System Prompts

Invest time in your system prompt:

```python
def get_system_prompt(cls):
    return """You are a data analysis expert.

## Core Principles
1. Always verify data quality
2. Use appropriate statistical methods
3. Visualize results clearly
4. Explain assumptions and limitations

## Process
1. Understand the question
2. Examine the data
3. Choose appropriate methods
4. Analyze thoroughly
5. Present clear results

...
"""
```

### 5. Test Thoroughly

Test both unit functionality and integration:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=my_vertical --cov-report=html
```

---

## Getting Help

- **SDK README**: See `README.md`
- **Contract Stability Policy**: See [CONTRACT_STABILITY.md](CONTRACT_STABILITY.md)
- **Migration Guide**: See `MIGRATION.md`
- **Examples**: Check `victor-contracts/examples/`
- **Repository external package example**: See `../examples/external_vertical/README.md`
- **Issues**: https://github.com/anvai-labs/victor/issues

---

## License

Apache-2.0
