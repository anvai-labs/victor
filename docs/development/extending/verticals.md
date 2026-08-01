# Vertical Development Guide

This guide covers how to develop custom verticals for Victor, enabling domain-specific AI assistants with specialized tools, prompts, and workflows.

## Overview

### What Are Verticals?

Verticals are domain-specific configurations that transform Victor into specialized AI assistants. Each vertical defines:

- **Tools**: Which tools the assistant can use
- **System Prompt**: Domain expertise and behavioral guidelines
- **Stages**: Workflow stages for task progression
- **Extensions**: Middleware, safety checks, and integrations

Victor includes five built-in verticals:

| Vertical | Description | Use Case |
|----------|-------------|----------|
| `coding` | Software development assistant | Code exploration, bug fixing, feature implementation |
| `research` | Web research and synthesis | Fact-checking, literature review, report generation |
| `devops` | Infrastructure and deployment | Docker, Terraform, CI/CD pipelines |
| `data_analysis` | Data exploration and visualization | Pandas, statistics, charts |
| `rag` | Document retrieval and Q&A | Knowledge base queries, document search |

### Why Create Custom Verticals?

Custom verticals enable you to:

1. **Restrict tool access** - Limit to only relevant tools for safety and focus
2. **Provide domain expertise** - System prompts with specialized knowledge
3. **Define workflows** - Multi-step processes for complex tasks
4. **Integrate extensions** - Custom middleware, safety checks, and handlers

### Authoring Modes

Victor is in a migration period with two authoring styles:

| Use case | Recommended surface | Notes |
|----------|---------------------|-------|
| External/package verticals | `victor_contracts` | Preferred path for new work. Use `VerticalBase`, `ToolNames`, `CapabilityIds`, and `get_definition()` |
| Bundled/runtime-integrated verticals | `victor.core` / `victor.framework` | Legacy/internal path while runtime boundaries are being cleaned up |

If you are publishing a standalone vertical package, depend on `victor-contracts`
only. The runtime-specific sections later in this guide still describe legacy
core-integrated extension points and remain relevant mainly for bundled
victor-ai verticals during migration.

For step-by-step migration from the legacy core-coupled pattern to the
contract-first surface, see `victor-contracts/MIGRATION.md` in the repository.

For a complete standalone package example that follows the contract-only authoring
model and runtime entry-point discovery, see `examples/external_vertical/README.md`
in the repository (the `victor-security` package referenced from
[CONTRIBUTING.md](https://github.com/anvai-labs/victor/blob/main/CONTRIBUTING.md)).

## VerticalBase Interface

External verticals should inherit from the contract `VerticalBase` and implement the
definition-layer hooks that feed `get_definition()`:

```python
from victor_contracts import ToolNames, VerticalBase

class MyVertical(VerticalBase):
    name = "my_vertical"  # Unique identifier
    description = "Description of what this vertical does"

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tools(cls) -> list[str]:
        """Return list of tool names available to this vertical."""
        return [ToolNames.READ, ToolNames.WRITE, ToolNames.GREP]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return the system prompt for this vertical."""
        return "You are an expert assistant specializing in..."
```

### Required Members

| Member | Type | Description |
|--------|------|-------------|
| `name` | `str` | Unique identifier for the vertical (lowercase, underscores) |
| `get_name()` | `classmethod` | Returns the canonical vertical identifier |
| `get_description()` | `classmethod` | Returns the human-readable description |
| `get_tools()` | `classmethod` | Returns list of tool names to enable |
| `get_system_prompt()` | `classmethod` | Returns the system prompt text |

### Optional Members

| Member | Default | Description |
|--------|---------|-------------|
| `description` | `""` | Human-readable description |
| `version` | `"1.0.0"` | Semantic version of the vertical |
| `get_stages()` | 7-stage workflow | Stage definitions for task progression |
| `get_definition()` | Derived from hooks | Returns validated SDK manifest data |
| `get_tool_requirements()` | `get_tools()` | Typed tool requirements for manifest-first authoring |
| `get_capability_requirements()` | `[]` | Typed runtime capability requirements |
| `get_team_declarations()` | `{}` | Declarative team layouts for contract-first authoring |
| `get_team_metadata()` | Derived from hooks | Typed team metadata bridge for `get_definition()` |
| `get_provider_hints()` | Default hints | Provider selection preferences |
| `get_evaluation_criteria()` | Basic criteria | Quality evaluation criteria |
| `get_middleware()` | `[]` | Middleware implementations |
| `get_safety_extension()` | `None` | Safety check patterns |
| `get_prompt_contributor()` | `None` | Task-type prompt hints |
| `get_mode_config_provider()` | `None` | Operational modes (fast/thorough) |
| `get_tool_dependency_provider()` | `None` | Tool execution patterns |
| `get_workflow_provider()` | `None` | YAML workflow provider |
| `get_handlers()` | `{}` | Compute handlers for workflows |
| `get_tool_graph()` | `None` | Tool execution graph |
| `get_team_specs()` | `None` | Legacy/runtime team configuration fallback |

## Creating a Custom Vertical

The walkthrough below shows a runtime-integrated vertical inside the Victor
repository. For standalone packages, use the contract-only pattern shown in the
sections above and in `victor-contracts/README.md`.

### Step 1: Define the Vertical Class

Create a new Python file for your vertical:

```python
# victor_security/assistant.py
from typing import Any, Dict, List, Optional
from victor_contracts import StageDefinition, VerticalBase

class SecurityAssistant(VerticalBase):
    """Security analysis and vulnerability assessment assistant."""

    name = "security"
    description = "Security analysis, vulnerability assessment, and code auditing"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Tools for security analysis."""
        return [
            # Core filesystem (read-only focused)
            "read",
            "ls",
            "grep",
            "overview",
            # Code analysis
            "code_search",
            "symbol",
            "refs",
            # Shell for running security tools
            "shell",
            # Web for CVE lookups
            "web_search",
            "web_fetch",
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Security-focused system prompt."""
        return """You are a security analyst assistant specializing in:
- Vulnerability assessment and code auditing
- Security best practices and OWASP guidelines
- Dependency vulnerability analysis
- Authentication and authorization review
- Secrets and sensitive data detection

Guidelines:
1. Always analyze code with a security-first mindset
2. Check for common vulnerabilities (SQLi, XSS, CSRF, etc.)
3. Review authentication and session management
4. Identify hardcoded secrets or sensitive data
5. Suggest secure coding practices
6. Reference CVE databases when applicable

Be thorough but prioritize high-severity issues."""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Security-specific workflow stages."""
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the security assessment scope",
                tools={"read", "ls", "overview"},
                keywords=["audit", "assess", "check", "review", "scan"],
                next_stages={"RECONNAISSANCE", "ANALYSIS"},
            ),
            "RECONNAISSANCE": StageDefinition(
                name="RECONNAISSANCE",
                description="Gathering information about the target",
                tools={"grep", "code_search", "overview", "symbol"},
                keywords=["find", "search", "discover", "enumerate"],
                next_stages={"ANALYSIS", "VULNERABILITY_SCAN"},
            ),
            "VULNERABILITY_SCAN": StageDefinition(
                name="VULNERABILITY_SCAN",
                description="Scanning for known vulnerabilities",
                tools={"shell", "grep", "web_search"},
                keywords=["scan", "detect", "identify", "vulnerability"],
                next_stages={"ANALYSIS", "REPORTING"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Analyzing findings and assessing risk",
                tools={"read", "refs", "symbol", "web_fetch"},
                keywords=["analyze", "risk", "severity", "impact"],
                next_stages={"REPORTING", "REMEDIATION"},
            ),
            "REMEDIATION": StageDefinition(
                name="REMEDIATION",
                description="Suggesting and implementing fixes",
                tools={"read", "grep", "shell"},
                keywords=["fix", "patch", "remediate", "mitigate"],
                next_stages={"VERIFICATION", "REPORTING"},
            ),
            "VERIFICATION": StageDefinition(
                name="VERIFICATION",
                description="Verifying fixes are effective",
                tools={"shell", "grep", "read"},
                keywords=["verify", "test", "confirm", "validate"],
                next_stages={"REPORTING"},
            ),
            "REPORTING": StageDefinition(
                name="REPORTING",
                description="Generating security report",
                tools={"read"},
                keywords=["report", "summarize", "document", "findings"],
                next_stages=set(),
            ),
        }

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Provider preferences for security tasks."""
        return {
            "preferred_providers": ["anthropic", "openai"],
            "min_context_window": 100000,
            "requires_tool_calling": True,
            "prefers_extended_thinking": True,  # Security needs careful analysis
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Criteria for evaluating security analysis."""
        return [
            "Vulnerability detection accuracy",
            "Risk assessment quality",
            "Coverage of OWASP Top 10",
            "False positive rate",
            "Remediation guidance quality",
            "Report clarity and completeness",
        ]
```

### Step 2: Create the Package Structure

For new contract-first packages, prefer the target split linked above. The example
below shows a standalone package layout with a contract definition layer and
optional runtime-side helpers.

Organize your vertical as a package:

```
victor-security/
  victor_security/
    __init__.py
    assistant.py          # SecurityAssistant class
    safety.py             # Safety extension (optional)
    prompts.py            # Prompt contributor (optional)
    mode_config.py        # Mode configurations (optional)
    escape_hatches.py     # YAML workflow conditions (optional)
    workflows/
        __init__.py
        security_audit.yaml
```

### Step 3: Register the Vertical

Publish a thin plugin wrapper and register the vertical through the plugin context:

```python
from victor_contracts import PluginContext, VictorPlugin

from victor_security.assistant import SecurityAssistant


class SecurityPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "security"

    def register(self, context: PluginContext) -> None:
        context.register_vertical(SecurityAssistant)


plugin = SecurityPlugin()
```

### Step 4: Use the Vertical

```python
from victor_security.assistant import SecurityAssistant

# Get configuration
config = SecurityAssistant.get_config()
print(f"Tools: {config.tools.get_tool_names()}")

# Get extensions
extensions = SecurityAssistant.get_extensions()

# Create an agent with this vertical
agent = await Agent.create(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    tools=config.tools,
    system_prompt=config.system_prompt,
)
```

## YAML Workflow Integration

Verticals can define complex workflows using YAML with Python escape hatches for conditions that cannot be expressed declaratively.

### Workflow Directory Structure

```
victor/security/
    workflows/
        __init__.py           # WorkflowProvider implementation
        security_audit.yaml   # Workflow definition
    escape_hatches.py         # Python conditions and transforms
```

### Creating a YAML Workflow

```yaml
# victor/security/workflows/security_audit.yaml
workflows:
  security_audit:
    description: "Comprehensive security audit workflow"
    metadata:
      version: "1.0"
      vertical: "security"

    nodes:
      - id: gather_context
        type: agent
        role: security_analyst
        goal: "Understand the codebase structure and identify entry points"
        tool_budget: 15
        output: context_report
        next: [identify_vulnerabilities]

      - id: identify_vulnerabilities
        type: agent
        role: vulnerability_scanner
        goal: "Scan for common vulnerabilities using gathered context"
        tool_budget: 20
        inputs:
          context: $ctx.context_report
        output: vulnerability_list
        next: [assess_severity]

      - id: assess_severity
        type: compute
        handler: severity_assessment  # Maps to escape hatch
        inputs:
          vulnerabilities: $ctx.vulnerability_list
        output: prioritized_findings
        next: [check_findings]

      - id: check_findings
        type: condition
        condition: "has_critical_findings"  # Escape hatch function
        branches:
          "critical": deep_analysis
          "non_critical": generate_report

      - id: deep_analysis
        type: agent
        role: security_expert
        goal: "Deep dive into critical vulnerabilities"
        tool_budget: 30
        inputs:
          findings: $ctx.prioritized_findings
        output: detailed_analysis
        next: [generate_report]

      - id: generate_report
        type: agent
        role: report_writer
        goal: "Generate comprehensive security report"
        tool_budget: 10
        output: final_report
        next: []
```

### Escape Hatches for Complex Logic

```python
# victor/security/escape_hatches.py
"""Escape hatches for Security YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.
"""

from typing import Any, Dict

# =============================================================================
# Condition Functions
# =============================================================================

def has_critical_findings(ctx: Dict[str, Any]) -> str:
    """Check if any findings are critical severity.

    Args:
        ctx: Workflow context with keys:
            - prioritized_findings (list): List of findings with severity

    Returns:
        "critical" or "non_critical"
    """
    findings = ctx.get("prioritized_findings", [])

    for finding in findings:
        if finding.get("severity") in ("critical", "high"):
            return "critical"

    return "non_critical"


def vulnerability_count_check(ctx: Dict[str, Any]) -> str:
    """Determine next action based on vulnerability count.

    Args:
        ctx: Workflow context with vulnerability_list

    Returns:
        "many", "few", or "none"
    """
    vulns = ctx.get("vulnerability_list", [])
    count = len(vulns)

    if count == 0:
        return "none"
    elif count <= 5:
        return "few"
    else:
        return "many"


# =============================================================================
# Transform Functions
# =============================================================================

def merge_findings(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge findings from multiple analysis passes.

    Args:
        ctx: Workflow context with multiple finding lists

    Returns:
        Merged and deduplicated findings
    """
    all_findings = []

    for key in ["static_findings", "dynamic_findings", "dependency_findings"]:
        findings = ctx.get(key, [])
        all_findings.extend(findings)

    # Deduplicate by vulnerability ID
    seen = set()
    unique = []
    for finding in all_findings:
        vuln_id = finding.get("id")
        if vuln_id and vuln_id not in seen:
            seen.add(vuln_id)
            unique.append(finding)

    return {
        "total_findings": len(unique),
        "findings": unique,
        "sources": ["static", "dynamic", "dependency"],
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    "has_critical_findings": has_critical_findings,
    "vulnerability_count_check": vulnerability_count_check,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    "merge_findings": merge_findings,
}
```

### Implementing WorkflowProviderProtocol

```python
# victor/security/workflows/__init__.py
"""Security workflow provider."""

from pathlib import Path
from typing import List, Tuple

from victor.framework.workflows.base_yaml_provider import BaseYAMLWorkflowProvider


class SecurityWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides security-specific YAML workflows.

    Workflows are loaded from victor/security/workflows/*.yaml
    with escape hatches from victor/security/escape_hatches.py
    """

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for escape hatches."""
        return "victor.security.escape_hatches"

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns."""
        return [
            (r"security\s+audit", "security_audit"),
            (r"vulnerability\s+scan", "vulnerability_scan"),
            (r"check\s+for\s+vulnerabilities", "quick_scan"),
        ]

    def get_workflow_for_task_type(self, task_type: str) -> str | None:
        """Map task types to workflow names."""
        mapping = {
            "audit": "security_audit",
            "scan": "vulnerability_scan",
            "pentest": "penetration_test",
        }
        return mapping.get(task_type.lower())
```

Then connect it to your vertical:

```python
# In SecurityAssistant class
@classmethod
def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
    """Get security-specific workflow provider."""
    from victor.security.workflows import SecurityWorkflowProvider
    return SecurityWorkflowProvider()
```

## External Plugin Registration

External packages should register runtime integrations through the `victor.plugins`
entry point. That gives the package a `PluginContext` with access to framework
registration hooks for verticals, commands, tools, chunkers, and custom workflow
node executors.

### Package Structure

```
victor-security/
    pyproject.toml
    src/victor_security/
        __init__.py
        assistant.py
        prompts.py
        victor-vertical.toml
        tool_dependencies.py
        runtime/
            __init__.py
            workflows/
                __init__.py
                security_audit.yaml
            escape_hatches.py
```

### Entry Point Configuration

```toml
# pyproject.toml
[project]
name = "victor-security"
version = "1.0.0"
dependencies = ["victor-contracts>=1.0.0"]

[project.optional-dependencies]
runtime = ["victor-ai>=0.4.0"]

[project.entry-points."victor.plugins"]
security = "victor_security:plugin"
```

### Package Implementation

```python
# victor_security/__init__.py
from victor_security.plugin import plugin

__all__ = ["plugin"]
```

```python
# victor_security/plugin.py
from victor_contracts import PluginContext, VictorPlugin
from victor_security.assistant import SecurityAssistant


class SecurityPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "security"

    def register(self, context: PluginContext) -> None:
        context.register_vertical(SecurityAssistant)


plugin = SecurityPlugin()
```

```python
# victor_security/assistant.py
from victor_contracts import ToolNames, VerticalBase


class SecurityAssistant(VerticalBase):
    name = "security"
    description = "Security analysis and vulnerability assessment"

    @classmethod
    def get_tools(cls) -> list[str]:
        return [
            ToolNames.READ,
            ToolNames.GREP,
            ToolNames.SHELL,
            ToolNames.WEB_SEARCH,
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a security analyst..."
```

### Registering Custom Workflow Node Executors

If your package introduces a custom workflow node type, register its executor
through the same plugin context instead of patching the core workflow factory.

```python
# victor_security/plugin.py
from victor_contracts import PluginContext, VictorPlugin
from victor.workflows import CompiledGraphNodeResult


class ThreatModelExecutor:
    def __init__(self, container=None):
        self._container = container

    async def execute(self, node, state):
        result = {"threat_model": f"modeled:{node.id}"}
        state.setdefault("_node_results", {})[node.id] = CompiledGraphNodeResult(
            node_id=node.id,
            success=True,
            output=result,
            metadata={"node_type": "threat_model"},
        )
        state.update(result)
        return state


class SecurityPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "security"

    def register(self, context: PluginContext) -> None:
        context.register_vertical(SecurityAssistant)
        context.register_workflow_node_executor("threat_model", ThreatModelExecutor)
```

For application-local extensions that are not packaged as plugins, the same
registration helpers are also available from `victor.workflows`:

```python
from victor.workflows import register_workflow_node_executor


register_workflow_node_executor("threat_model", ThreatModelExecutor)
```

### Installation and Discovery

```bash
# Install the external package
pip install victor-security

# Or install with runtime helpers if the package ships them
pip install "victor-security[runtime]"

# Victor automatically discovers and registers the plugin
victor chat --vertical security
```

`victor.verticals` remains a legacy compatibility entry point for definition-only
packages, but new runtime-integrated extensions should use `victor.plugins`.

### Validation Requirements

External verticals must meet these requirements to be registered:

1. **Inherit from VerticalBase**: The class must be a subclass of `VerticalBase`
2. **Define `name` attribute**: A non-empty string identifier
3. **Implement `get_tools()`**: Must return a `List[str]`
4. **Implement `get_system_prompt()`**: Must return a `str`

Victor validates these during discovery and logs warnings for invalid verticals.

## Testing Verticals

### Unit Testing

```python
# tests/unit/security/test_security_assistant.py
import pytest
from typing import List

from victor_contracts import StageDefinition, VerticalConfig
from victor_contracts.testing import MockPluginContext


class TestSecurityAssistant:
    """Tests for SecurityAssistant vertical."""

    @pytest.fixture
    def vertical(self):
        """Get the security vertical."""
        from victor_security.assistant import SecurityAssistant
        return SecurityAssistant

    def test_name_and_description(self, vertical):
        """Vertical should have name and description."""
        assert vertical.name == "security"
        assert "security" in vertical.description.lower()

    def test_get_tools_returns_list(self, vertical):
        """get_tools should return a list of strings."""
        tools = vertical.get_tools()
        assert isinstance(tools, list)
        assert all(isinstance(t, str) for t in tools)
        # Verify essential tools
        assert "read" in tools
        assert "grep" in tools

    def test_get_system_prompt(self, vertical):
        """get_system_prompt should return security-focused prompt."""
        prompt = vertical.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "security" in prompt.lower()

    def test_get_stages(self, vertical):
        """get_stages should return valid stage definitions."""
        stages = vertical.get_stages()
        assert isinstance(stages, dict)

        # Verify stage structure
        for name, stage in stages.items():
            assert isinstance(stage, StageDefinition)
            assert stage.name == name
            assert stage.description

    def test_get_config(self, vertical):
        """get_config should return complete configuration."""
        config = vertical.get_config()

        assert isinstance(config, VerticalConfig)
        assert config.system_prompt is not None
        assert len(config.stages) > 0
        assert config.metadata["vertical_name"] == "security"

    def test_get_extensions_never_returns_none(self, vertical):
        """get_extensions should always return VerticalExtensions."""
        from victor_contracts import VerticalExtensions

        extensions = vertical.get_extensions()

        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)


class TestSecurityPluginRegistration:
    """Tests for plugin-context registration."""

    def test_registers_vertical(self):
        """Plugin should register the vertical with the host context."""
        from victor_security import SecurityPlugin
        from victor_security.assistant import SecurityAssistant

        context = MockPluginContext()
        SecurityPlugin().register(context)

        assert context.verticals["security"] is SecurityAssistant


class TestSecurityEscapeHatches:
    """Tests for escape hatch functions."""

    def test_has_critical_findings_with_critical(self):
        """Should return 'critical' when critical findings exist."""
        from victor.security.escape_hatches import has_critical_findings

        ctx = {
            "prioritized_findings": [
                {"id": "CVE-2024-1234", "severity": "critical"},
                {"id": "CVE-2024-5678", "severity": "low"},
            ]
        }

        result = has_critical_findings(ctx)
        assert result == "critical"

    def test_has_critical_findings_without_critical(self):
        """Should return 'non_critical' when no critical findings."""
        from victor.security.escape_hatches import has_critical_findings

        ctx = {
            "prioritized_findings": [
                {"id": "CVE-2024-9999", "severity": "low"},
            ]
        }

        result = has_critical_findings(ctx)
        assert result == "non_critical"
```

### Integration Testing

```python
# tests/integration/security/test_security_workflow.py
import pytest

from victor.core.verticals import get_vertical_loader


@pytest.mark.integration
class TestSecurityWorkflowIntegration:
    """Integration tests for security workflows."""

    def test_load_security_vertical(self):
        """Security vertical should load correctly."""
        loader = get_vertical_loader()
        loader.load("security")

        assert loader.active_vertical_name == "security"

        tools = loader.get_tools()
        assert "read" in tools
        assert "grep" in tools

    @pytest.mark.asyncio
    async def test_workflow_execution(self, mock_orchestrator):
        """Security audit workflow should execute."""
        from victor.security.workflows import SecurityWorkflowProvider

        provider = SecurityWorkflowProvider()
        workflow = provider.get_workflow("security_audit")

        assert workflow is not None
        assert workflow.name == "security_audit"
```

### Test Fixtures

Contract-first vertical packages usually need only package-local fixtures. If a
test needs to validate plugin registration, use `MockPluginContext` instead of
mutating the runtime registry:

```python
# conftest.py
import pytest

from victor_contracts.testing import MockPluginContext


@pytest.fixture
def plugin_context():
    """Provide an isolated plugin registration context."""
    return MockPluginContext()
```

## Advanced Topics

### Extension Caching

Verticals use caching to avoid repeated computation:

```python
@classmethod
def get_safety_extension(cls) -> Optional[SafetyExtensionProtocol]:
    """Get safety extension with caching."""
    def _create() -> SafetyExtensionProtocol:
        from victor.security.safety import SecuritySafetyExtension
        return SecuritySafetyExtension()

    return cls._get_cached_extension("safety_extension", _create)
```

Clear caches when needed:

```python
# Clear caches for this vertical
SecurityAssistant.clear_config_cache()

# Clear all vertical caches
SecurityAssistant.clear_config_cache(clear_all=True)
```

### Strict Extension Loading

Enable strict mode to fail fast on extension errors:

```python
class SecurityAssistant(VerticalBase):
    name = "security"
    strict_extension_loading = True  # Raise on any extension failure
    required_extensions = {"safety"}  # These must load successfully
```

### Tiered Tool Configuration

Define tool tiers for intelligent selection:

```python
@classmethod
def get_tiered_tool_config(cls) -> Optional[TieredToolConfig]:
    """Configure tool tiers for security analysis."""
    from victor.core.vertical_types import TieredToolConfig

    return TieredToolConfig(
        # Always included
        mandatory={"read", "grep", "ls"},
        # Essential for security
        vertical_core={"shell", "code_search", "web_search"},
        # Selected based on context
        semantic_pool={"symbol", "refs", "web_fetch"},
        # Stage-specific tools
        stage_tools={
            "VULNERABILITY_SCAN": {"shell", "web_search"},
            "ANALYSIS": {"read", "refs", "symbol"},
        },
        readonly_only_for_analysis=True,
    )
```

### Multi-Agent Teams

Define team configurations for complex tasks:

```python
@classmethod
def get_team_spec_provider(cls) -> Optional[TeamSpecProviderProtocol]:
    """Get security team specifications."""
    from victor.security.teams import SecurityTeamSpecProvider
    return SecurityTeamSpecProvider()
```

## Best Practices

1. **Start minimal**: Begin with essential tools and expand as needed
2. **Focus the prompt**: Write clear, domain-specific system prompts
3. **Define clear stages**: Map your domain's workflow to stages
4. **Use escape hatches wisely**: Keep YAML for structure, Python for logic
5. **Test thoroughly**: Unit test all components, integration test workflows
6. **Cache extensions**: Use `_get_cached_extension()` for expensive operations
7. **Handle errors gracefully**: Use non-strict mode with `required_extensions`
8. **Document your vertical**: Include docstrings and usage examples

## Reference

### Tool Names

Use canonical tool names and capability IDs from `victor_contracts`:

```python
from victor_contracts import CapabilityIds, ToolNames

tools = [
    ToolNames.READ,        # "read"
    ToolNames.WRITE,       # "write"
    ToolNames.EDIT,        # "edit"
    ToolNames.GREP,        # "grep"
    ToolNames.SHELL,       # "shell"
    ToolNames.GIT,         # "git"
    ToolNames.WEB_SEARCH,  # "web_search"
    ToolNames.WEB_FETCH,   # "web_fetch"
]

capabilities = [
    CapabilityIds.FILE_OPS,   # "file_ops"
    CapabilityIds.GIT,        # "git"
    CapabilityIds.WEB_ACCESS, # "web_access"
]
```

### Stage Transitions

Stages form a directed graph. Define valid transitions:

```python
StageDefinition(
    name="ANALYSIS",
    description="Analyzing findings",
    next_stages={"REMEDIATION", "REPORTING"},  # Valid next stages
)
```

### Extension Protocols

Available extension protocols:

| Protocol | Method | Purpose |
|----------|--------|---------|
| `MiddlewareProtocol` | `get_middleware()` | Tool execution processing |
| `SafetyExtensionProtocol` | `get_safety_extension()` | Dangerous operation patterns |
| `PromptContributorProtocol` | `get_prompt_contributor()` | Task-type hints |
| `ModeConfigProviderProtocol` | `get_mode_config_provider()` | Operational modes |
| `ToolDependencyProviderProtocol` | `get_tool_dependency_provider()` | Tool execution patterns |
| `WorkflowProviderProtocol` | `get_workflow_provider()` | YAML workflows |
| `TeamSpecProviderProtocol` | `get_team_spec_provider()` | Multi-agent teams |
| `RLConfigProviderProtocol` | `get_rl_config_provider()` | RL configurations |
| `EnrichmentStrategyProtocol` | `get_enrichment_strategy()` | Prompt enrichment |
