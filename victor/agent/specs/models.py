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

"""Agent specification DSL.

Provides type-safe, composable agent definitions with:
- Declarative specifications
- Capability and constraint modeling
- Model preference hints
- Serialization to/from YAML/JSON
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import hashlib

from victor.tools.core_tool_aliases import canonicalize_core_tool_name


def _canonicalize_tool_set(tools: Set[str]) -> Set[str]:
    """Normalize tool names onto the canonical internal registry surface."""
    return {canonicalize_core_tool_name(tool) for tool in tools if tool}


class ModelPreference(str, Enum):
    """Model selection preferences.

    Instead of hardcoding model names, agents declare what
    kind of model they need. The system selects the best
    available model matching the preference.
    """

    # Task-based preferences
    REASONING = "reasoning"  # Complex analysis, planning
    CODING = "coding"  # Code generation, review
    FAST = "fast"  # Quick responses, low latency
    BALANCED = "balanced"  # Good all-around performance
    CREATIVE = "creative"  # Writing, brainstorming

    # Cost-based preferences
    BUDGET = "budget"  # Cheapest option
    PREMIUM = "premium"  # Best quality regardless of cost

    # Capability-based
    TOOL_USE = "tool_use"  # Strong tool calling
    LONG_CONTEXT = "long_context"  # Large context window

    # Default
    DEFAULT = "default"  # Use orchestrator's default


class DelegationPolicy(str, Enum):
    """How an agent can delegate to others."""

    NONE = "none"  # No delegation allowed
    ASK = "ask"  # Ask user before delegating
    AUTO = "auto"  # Automatically delegate when appropriate
    HIERARCHICAL = "hierarchical"  # Only delegate to subordinates


class OutputFormat(str, Enum):
    """Expected output format from agent."""

    TEXT = "text"  # Free-form text
    JSON = "json"  # Structured JSON
    CODE = "code"  # Source code
    MARKDOWN = "markdown"  # Markdown document
    STRUCTURED = "structured"  # Pydantic model


@dataclass
class AgentCapabilities:
    """What an agent can do.

    Defines tools, skills, and behavioral capabilities.
    """

    # Tool access
    tools: Set[str] = field(default_factory=set)
    tool_patterns: List[str] = field(default_factory=list)  # Glob patterns

    # Skills (high-level capabilities)
    skills: Set[str] = field(default_factory=set)

    # Behavioral flags
    can_delegate: bool = False
    can_ask_user: bool = True
    can_browse_web: bool = False
    can_execute_code: bool = False
    can_modify_files: bool = False

    def __post_init__(self) -> None:
        """Normalize explicit tool names to canonical runtime identifiers."""
        self.tools = _canonicalize_tool_set(self.tools)

    def resolve_skills(self, registry: Any) -> List[Any]:
        """Resolve skill names to SkillDefinition objects.

        Args:
            registry: A SkillRegistry instance

        Returns:
            List of resolved SkillDefinition objects for skills in this set
        """
        resolved = []
        for skill_name in sorted(self.skills):
            skill = registry.get_optional(skill_name)
            if skill is not None:
                resolved.append(skill)
        return resolved

    def allows_tool(self, tool_name: str) -> bool:
        """Check if a tool is allowed."""
        canonical_tool_name = canonicalize_core_tool_name(tool_name)
        if canonical_tool_name in self.tools:
            return True
        # Check patterns (simple glob)
        import fnmatch

        for pattern in self.tool_patterns:
            if fnmatch.fnmatch(canonical_tool_name, pattern) or fnmatch.fnmatch(tool_name, pattern):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tools": list(self.tools),
            "tool_patterns": self.tool_patterns,
            "skills": list(self.skills),
            "can_delegate": self.can_delegate,
            "can_ask_user": self.can_ask_user,
            "can_browse_web": self.can_browse_web,
            "can_execute_code": self.can_execute_code,
            "can_modify_files": self.can_modify_files,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapabilities":
        """Deserialize from dictionary."""
        return cls(
            tools=set(data.get("tools", [])),
            tool_patterns=data.get("tool_patterns", []),
            skills=set(data.get("skills", [])),
            can_delegate=data.get("can_delegate", False),
            can_ask_user=data.get("can_ask_user", True),
            can_browse_web=data.get("can_browse_web", False),
            can_execute_code=data.get("can_execute_code", False),
            can_modify_files=data.get("can_modify_files", False),
        )


@dataclass
class AgentConstraints:
    """Limits and requirements for agent execution.

    Inspired by Kubernetes resource limits.
    """

    # Token limits
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None

    # Iteration limits
    max_iterations: int = 50
    max_tool_calls: int = 100

    # Cost limits
    max_cost_usd: Optional[float] = None

    # Time limits
    timeout_seconds: Optional[float] = None

    # Context requirements
    min_context_tokens: Optional[int] = None
    required_tools: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Normalize required tools to canonical runtime identifiers."""
        self.required_tools = _canonicalize_tool_set(self.required_tools)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "max_cost_usd": self.max_cost_usd,
            "timeout_seconds": self.timeout_seconds,
            "min_context_tokens": self.min_context_tokens,
            "required_tools": list(self.required_tools),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConstraints":
        """Deserialize from dictionary."""
        return cls(
            max_input_tokens=data.get("max_input_tokens"),
            max_output_tokens=data.get("max_output_tokens"),
            max_total_tokens=data.get("max_total_tokens"),
            max_iterations=data.get("max_iterations", 50),
            max_tool_calls=data.get("max_tool_calls", 100),
            max_cost_usd=data.get("max_cost_usd"),
            timeout_seconds=data.get("timeout_seconds"),
            min_context_tokens=data.get("min_context_tokens"),
            required_tools=set(data.get("required_tools", [])),
        )


@dataclass
class AgentSpec:
    """Declarative agent specification.

    This is the core DSL element for defining agents.
    Supports both Python construction and YAML/JSON loading.

    Example:
        # Python DSL
        agent = AgentSpec(
            name="researcher",
            description="Finds and analyzes information",
            capabilities=AgentCapabilities(
                tools={"web_search", "read"},
                can_browse_web=True,
            ),
            model_preference=ModelPreference.REASONING,
        )

        # Or from YAML
        agent = AgentSpec.from_dict({
            "name": "researcher",
            "description": "Finds and analyzes information",
            "capabilities": ["web_search", "read"],
            "model_preference": "reasoning",
        })
    """

    # Identity
    name: str
    description: str

    # Capabilities (what it can do)
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)

    # Constraints (limits)
    constraints: AgentConstraints = field(default_factory=AgentConstraints)

    # Model preferences
    model_preference: ModelPreference = ModelPreference.DEFAULT
    model_hints: Dict[str, Any] = field(default_factory=dict)

    # Behavioral settings
    system_prompt: Optional[str] = None
    output_format: OutputFormat = OutputFormat.TEXT
    delegation_policy: DelegationPolicy = DelegationPolicy.NONE

    # Metadata
    version: str = "1.0"
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique agent ID."""
        content = f"{self.name}:{self.version}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"agent_{hash_val}"

    def with_capabilities(
        self,
        tools: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> "AgentSpec":
        """Create new spec with updated capabilities.

        Args:
            tools: Additional tools to allow
            **kwargs: Other capability overrides

        Returns:
            New AgentSpec with merged capabilities
        """
        new_caps = AgentCapabilities(
            tools=self.capabilities.tools | (tools or set()),
            tool_patterns=self.capabilities.tool_patterns + kwargs.pop("tool_patterns", []),
            skills=self.capabilities.skills | set(kwargs.pop("skills", [])),
            can_delegate=kwargs.pop("can_delegate", self.capabilities.can_delegate),
            can_ask_user=kwargs.pop("can_ask_user", self.capabilities.can_ask_user),
            can_browse_web=kwargs.pop("can_browse_web", self.capabilities.can_browse_web),
            can_execute_code=kwargs.pop("can_execute_code", self.capabilities.can_execute_code),
            can_modify_files=kwargs.pop("can_modify_files", self.capabilities.can_modify_files),
        )
        return AgentSpec(
            name=self.name,
            description=self.description,
            capabilities=new_caps,
            constraints=self.constraints,
            model_preference=self.model_preference,
            model_hints=self.model_hints,
            system_prompt=self.system_prompt,
            output_format=self.output_format,
            delegation_policy=self.delegation_policy,
            version=self.version,
            tags=self.tags,
            metadata=self.metadata,
        )

    def with_constraints(self, **kwargs: Any) -> "AgentSpec":
        """Create new spec with updated constraints.

        Args:
            **kwargs: Constraint overrides

        Returns:
            New AgentSpec with merged constraints
        """
        new_constraints = AgentConstraints(
            max_input_tokens=kwargs.get("max_input_tokens", self.constraints.max_input_tokens),
            max_output_tokens=kwargs.get("max_output_tokens", self.constraints.max_output_tokens),
            max_total_tokens=kwargs.get("max_total_tokens", self.constraints.max_total_tokens),
            max_iterations=kwargs.get("max_iterations", self.constraints.max_iterations),
            max_tool_calls=kwargs.get("max_tool_calls", self.constraints.max_tool_calls),
            max_cost_usd=kwargs.get("max_cost_usd", self.constraints.max_cost_usd),
            timeout_seconds=kwargs.get("timeout_seconds", self.constraints.timeout_seconds),
            min_context_tokens=kwargs.get(
                "min_context_tokens", self.constraints.min_context_tokens
            ),
            required_tools=self.constraints.required_tools | set(kwargs.get("required_tools", [])),
        )
        return AgentSpec(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
            constraints=new_constraints,
            model_preference=self.model_preference,
            model_hints=self.model_hints,
            system_prompt=self.system_prompt,
            output_format=self.output_format,
            delegation_policy=self.delegation_policy,
            version=self.version,
            tags=self.tags,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (YAML/JSON compatible)."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities.to_dict(),
            "constraints": self.constraints.to_dict(),
            "model_preference": self.model_preference.value,
            "model_hints": self.model_hints,
            "system_prompt": self.system_prompt,
            "output_format": self.output_format.value,
            "delegation_policy": self.delegation_policy.value,
            "version": self.version,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpec":
        """Deserialize from dictionary.

        Supports shorthand notation for capabilities:
        - capabilities: ["tool1", "tool2"]  # List of tools
        - capabilities: {tools: [...], can_browse_web: true}  # Full spec
        """
        # Handle capabilities shorthand
        caps_data = data.get("capabilities", {})
        if isinstance(caps_data, list):
            # Shorthand: list of tool names
            capabilities = AgentCapabilities(tools=set(caps_data))
        elif isinstance(caps_data, dict):
            capabilities = AgentCapabilities.from_dict(caps_data)
        else:
            capabilities = AgentCapabilities()

        # Handle constraints
        constraints_data = data.get("constraints", {})
        if isinstance(constraints_data, dict):
            constraints = AgentConstraints.from_dict(constraints_data)
        else:
            constraints = AgentConstraints()

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            capabilities=capabilities,
            constraints=constraints,
            model_preference=ModelPreference(data.get("model_preference", "default")),
            model_hints=data.get("model_hints", {}),
            system_prompt=data.get("system_prompt"),
            output_format=OutputFormat(data.get("output_format", "text")),
            delegation_policy=DelegationPolicy(data.get("delegation_policy", "none")),
            version=data.get("version", "1.0"),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"AgentSpec(name={self.name!r}, "
            f"model={self.model_preference.value}, "
            f"tools={len(self.capabilities.tools)})"
        )
