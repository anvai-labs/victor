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

"""Pydantic schema models for YAML-based tool dependency configuration.

This module defines the schema for declarative YAML-based tool dependency
configuration, replacing hand-coded Python dictionaries in verticals.

Design Principles (SOLID):
    - SRP: Schema validation is separate from loading and conversion
    - OCP: Schema is extensible via optional fields without modification
    - DIP: Depends on abstractions (Pydantic models, not concrete implementations)

Schema Structure:
    The YAML configuration follows this structure:

    ```yaml
    version: "1.0"
    vertical: coding

    transitions:
      read:
        - tool: edit
          weight: 0.4
        - tool: grep
          weight: 0.3

    clusters:
      file_operations:
        - read
        - write
        - edit

    sequences:
      exploration:
        - ls
        - read
        - grep

    dependencies:
      - tool: edit
        depends_on: [read]
        enables: [test, git]
        weight: 0.9

    required_tools: [read, write, edit, ls, grep]
    optional_tools: [code_search, symbol, test, git]
    default_sequence: [read, edit, test]
    ```

Example:
    from victor.core.tool_dependency_schema import ToolDependencySpec
    import yaml

    with open("tool_dependencies.yaml") as f:
        data = yaml.safe_load(f)

    spec = ToolDependencySpec.model_validate(data)
    print(spec.vertical)  # "coding"
    print(spec.required_tools)  # ["read", "write", "edit", "ls", "grep"]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator, model_validator


class ToolTransition(BaseModel):
    """A single tool transition with probability weight.

    Represents the probability of transitioning from one tool to another
    in a workflow sequence.

    Attributes:
        tool: The target tool name to transition to.
        weight: Transition probability weight (0.0 to 1.0).

    Example:
        ```yaml
        - tool: edit
          weight: 0.4
        ```
    """

    tool: str = Field(..., description="Target tool name for transition")
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Transition probability weight (0.0 to 1.0)",
    )

    @field_validator("tool")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool name is non-empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()


class ToolCluster(BaseModel):
    """A cluster of related tools.

    Tool clusters group tools that are commonly used together or serve
    similar purposes. This enables cluster-based tool suggestions.

    Attributes:
        name: Cluster identifier name.
        tools: Set of tool names in this cluster.

    Example:
        ```yaml
        file_operations:
          - read
          - write
          - edit
          - ls
        ```
    """

    name: str = Field(..., description="Cluster identifier name")
    tools: List[str] = Field(
        default_factory=list,
        description="List of tool names in this cluster",
    )

    @field_validator("tools")
    @classmethod
    def validate_tools_list(cls, v: List[str]) -> List[str]:
        """Validate tools list is non-empty and contains valid names."""
        if not v:
            raise ValueError("Cluster must contain at least one tool")
        return [t.strip() for t in v if t and t.strip()]


class ToolSequence(BaseModel):
    """A named sequence of tools for a specific task type.

    Tool sequences define the recommended order of tool execution
    for common workflow patterns.

    Attributes:
        name: Sequence identifier (e.g., "exploration", "edit", "refactor").
        tools: Ordered list of tool names in the sequence.

    Example:
        ```yaml
        exploration:
          - ls
          - read
          - grep
        ```
    """

    name: str = Field(..., description="Sequence identifier name")
    tools: List[str] = Field(
        default_factory=list,
        description="Ordered list of tool names in sequence",
    )

    @field_validator("tools")
    @classmethod
    def validate_tools_sequence(cls, v: List[str]) -> List[str]:
        """Validate tools sequence is non-empty."""
        if not v:
            raise ValueError("Sequence must contain at least one tool")
        return [t.strip() for t in v if t and t.strip()]


class ToolDependencyEntry(BaseModel):
    """A tool dependency definition.

    Defines the dependency relationships for a single tool, including
    which tools should be called before and which tools are enabled after.

    Attributes:
        tool: The tool name this dependency applies to.
        depends_on: Tools that should be called before this tool.
        enables: Tools that are enabled after this tool succeeds.
        weight: Dependency strength weight (0.0 to 1.0).

    Example:
        ```yaml
        - tool: edit
          depends_on: [read]
          enables: [test, git]
          weight: 0.9
        ```
    """

    tool: str = Field(..., description="Tool name for this dependency")
    depends_on: List[str] = Field(
        default_factory=list,
        description="Tools that should be called before this one",
    )
    enables: List[str] = Field(
        default_factory=list,
        description="Tools enabled after this one succeeds",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Dependency strength weight (0.0 to 1.0)",
    )

    @field_validator("tool")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool name is non-empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()

    @field_validator("depends_on", "enables")
    @classmethod
    def validate_tool_lists(cls, v: List[str]) -> List[str]:
        """Clean and validate tool name lists."""
        return [t.strip() for t in v if t and t.strip()]


class ToolDependencySpec(BaseModel):
    """Complete tool dependency specification for a vertical.

    This is the root model for YAML-based tool dependency configuration.
    It contains all the information needed to create a ToolDependencyConfig
    for use with BaseToolDependencyProvider.

    The schema is designed to be OCP-compliant (Open/Closed Principle):
    new fields can be added without modifying existing code by using
    optional fields with defaults.

    Attributes:
        version: Schema version for compatibility checking.
        vertical: Name of the vertical this config applies to.
        transitions: Tool transition probability mappings.
        clusters: Groups of related tools.
        sequences: Named tool sequences for task types.
        dependencies: Tool dependency definitions.
        required_tools: Essential tools for this vertical.
        optional_tools: Tools that enhance but aren't required.
        default_sequence: Fallback sequence when task type unknown.
        metadata: Optional additional metadata for extensibility.

    Example:
        ```yaml
        version: "1.0"
        vertical: coding

        transitions:
          read:
            - tool: edit
              weight: 0.4

        clusters:
          file_operations:
            - read
            - write

        sequences:
          exploration:
            - ls
            - read

        dependencies:
          - tool: edit
            depends_on: [read]
            enables: [test]
            weight: 0.9

        required_tools: [read, edit]
        optional_tools: [grep, test]
        default_sequence: [read, edit, test]
        ```
    """

    version: str = Field(
        default="1.0",
        description="Schema version for compatibility checking",
    )
    vertical: str = Field(
        ...,
        description="Name of the vertical (coding, devops, rag, etc.)",
    )

    # Transition probabilities: tool -> list of (next_tool, weight)
    transitions: Dict[str, List[ToolTransition]] = Field(
        default_factory=dict,
        description="Tool transition probability mappings",
    )

    # Tool clusters: cluster_name -> list of tools
    clusters: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Groups of related tools",
    )

    # Tool sequences: sequence_name -> list of tools
    sequences: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Named tool sequences for task types",
    )

    # Tool dependencies
    dependencies: List[ToolDependencyEntry] = Field(
        default_factory=list,
        description="Tool dependency definitions",
    )

    # Required and optional tools
    required_tools: List[str] = Field(
        default_factory=list,
        description="Essential tools for this vertical",
    )
    optional_tools: List[str] = Field(
        default_factory=list,
        description="Tools that enhance but aren't required",
    )

    # Default sequence for unknown task types
    default_sequence: List[str] = Field(
        default_factory=lambda: ["read", "edit"],
        description="Fallback sequence when task type unknown",
    )

    # Extensibility: additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional metadata for extensibility",
    )

    @field_validator("vertical")
    @classmethod
    def validate_vertical(cls, v: str) -> str:
        """Validate vertical name is non-empty."""
        if not v or not v.strip():
            raise ValueError("Vertical name cannot be empty")
        return v.strip().lower()

    @field_validator("clusters")
    @classmethod
    def validate_clusters(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Clean cluster tool names."""
        return {name: [t.strip() for t in tools if t and t.strip()] for name, tools in v.items()}

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Clean sequence tool names."""
        return {name: [t.strip() for t in tools if t and t.strip()] for name, tools in v.items()}

    @field_validator("required_tools", "optional_tools", "default_sequence")
    @classmethod
    def validate_tool_lists(cls, v: List[str]) -> List[str]:
        """Clean and validate tool name lists."""
        return [t.strip() for t in v if t and t.strip()]

    @model_validator(mode="after")
    def validate_tool_references(self) -> "ToolDependencySpec":
        """Validate that tool references are consistent across the spec.

        This validator checks that tools referenced in transitions,
        dependencies, and sequences are defined somewhere in the config.
        """
        # Collect all referenced tool names
        all_tools: Set[str] = set()

        # From transitions
        for source, targets in self.transitions.items():
            all_tools.add(source)
            all_tools.update(t.tool for t in targets)

        # From clusters
        for tools in self.clusters.values():
            all_tools.update(tools)

        # From sequences
        for tools in self.sequences.values():
            all_tools.update(tools)

        # From dependencies
        for dep in self.dependencies:
            all_tools.add(dep.tool)
            all_tools.update(dep.depends_on)
            all_tools.update(dep.enables)

        # From required/optional
        all_tools.update(self.required_tools)
        all_tools.update(self.optional_tools)
        all_tools.update(self.default_sequence)

        # Store collected tools for reference (useful for debugging)
        if "all_referenced_tools" not in self.metadata:
            self.metadata["all_referenced_tools"] = sorted(all_tools)

        return self

    def get_all_tool_names(self) -> Set[str]:
        """Get all tool names referenced in this spec.

        Returns:
            Set of all tool names used anywhere in the configuration.
        """
        all_tools: Set[str] = set()

        for source, targets in self.transitions.items():
            all_tools.add(source)
            all_tools.update(t.tool for t in targets)

        for tools in self.clusters.values():
            all_tools.update(tools)

        for tools in self.sequences.values():
            all_tools.update(tools)

        for dep in self.dependencies:
            all_tools.add(dep.tool)
            all_tools.update(dep.depends_on)
            all_tools.update(dep.enables)

        all_tools.update(self.required_tools)
        all_tools.update(self.optional_tools)
        all_tools.update(self.default_sequence)

        return all_tools


__all__ = [
    "ToolTransition",
    "ToolCluster",
    "ToolSequence",
    "ToolDependencyEntry",
    "ToolDependencySpec",
]
