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

"""Tool metadata models plus a deprecated registry compatibility wrapper."""

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)

if TYPE_CHECKING:
    from victor.tools.base import BaseTool


_LEGACY_METADATA_REGISTRY_WARNING_EMITTED = False


@dataclass
class ToolMetadata:
    """Semantic metadata for tool selection and discovery.

    This dataclass allows tools to define their own semantic information
    inline, enabling fully dynamic tool registration without needing to
    manually update configuration files or tool_selection.py.

    Attributes:
        category: Tool category (e.g., 'git', 'security', 'pipeline')
        keywords: Keywords that trigger this tool in user requests
        use_cases: High-level use cases for semantic matching
        examples: Example requests that should match this tool
        priority_hints: Usage hints for tool selection
        priority: Tool priority for selection (default: MEDIUM)
        access_mode: Access mode for approval tracking (default: READONLY)
        danger_level: Danger level for warnings (default: SAFE)
        stages: Conversation stages where this tool is relevant (e.g., ['reading', 'execution'])
        mandatory_keywords: Keywords that FORCE this tool to be included when matched.
                           Unlike regular keywords, these guarantee inclusion.
                           E.g., ["show diff", "compare"] for shell tool.
        task_types: Task types this tool is relevant for (analysis, action, generation,
                   search, edit, default). Used for classification-aware tool selection.
        signature_params: Parameters to track in loop detection signatures.
                         Only these parameters are included in the signature. Pagination
                         parameters (offset, limit, k, etc.) should be excluded.
                         E.g., ["path"] for read tool (excludes offset/limit).
        execution_category: Category for parallel execution and dependency analysis.
                           Determines which tools can safely run concurrently.
    """

    category: str = ""
    keywords: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    priority_hints: List[str] = field(default_factory=list)
    # Selection/approval metadata with sensible defaults
    priority: Optional["Priority"] = None  # Default: MEDIUM when None
    access_mode: Optional["AccessMode"] = None  # Default: READONLY when None
    danger_level: Optional["DangerLevel"] = None  # Default: SAFE when None
    # Stage affinity for conversation state machine
    stages: List[str] = field(default_factory=list)  # e.g., ["initial", "reading", "execution"]
    # NEW: Mandatory keyword triggers that force tool inclusion
    mandatory_keywords: List[str] = field(default_factory=list)  # e.g., ["show diff", "compare"]
    # NEW: Task types for classification-aware selection
    task_types: List[str] = field(default_factory=list)  # e.g., ["analysis", "search", "default"]
    # NEW: Signature parameters for loop detection
    signature_params: List[str] = field(
        default_factory=list
    )  # e.g., ["path"] (excludes offset/limit)
    # NEW: Execution category for parallel execution
    execution_category: Optional["ExecutionCategory"] = None  # Default: READ_ONLY when None
    # Per-argument semantic kind (FEP-0024): param-name -> ArgumentKind. Empty by default;
    # when declared, drives per-argument correction gating (only EXECUTABLE_CODE is
    # correctable; FILE_CONTENT is never auto-mutated).
    argument_kinds: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Apply defaults for None values to support backward compatibility."""
        if self.priority is None:
            self.priority = Priority.MEDIUM
        if self.access_mode is None:
            self.access_mode = AccessMode.READONLY
        if self.danger_level is None:
            self.danger_level = DangerLevel.SAFE
        if self.execution_category is None:
            self.execution_category = ExecutionCategory.READ_ONLY

    @property
    def is_critical(self) -> bool:
        """Check if this tool has CRITICAL priority."""
        return self.priority == Priority.CRITICAL

    @property
    def requires_approval(self) -> bool:
        """Check if this tool requires user approval."""
        return self.access_mode.requires_approval or self.danger_level.requires_confirmation

    @property
    def is_safe(self) -> bool:
        """Check if this tool is safe (readonly, no danger)."""
        return self.access_mode.is_safe and self.danger_level == DangerLevel.SAFE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for YAML/JSON export."""
        return {
            "category": self.category,
            "keywords": self.keywords,
            "use_cases": self.use_cases,
            "examples": self.examples,
            "priority_hints": self.priority_hints,
            "priority": self.priority.name if self.priority else "MEDIUM",
            "access_mode": self.access_mode.value if self.access_mode else "readonly",
            "danger_level": self.danger_level.value if self.danger_level else "safe",
            "stages": self.stages,
            "mandatory_keywords": self.mandatory_keywords,
            "task_types": self.task_types,
            "signature_params": self.signature_params,
            "execution_category": (
                self.execution_category.value if self.execution_category else "read_only"
            ),
        }

    @classmethod
    def from_contract(cls, contract: Any, tool: Any) -> "ToolMetadata":
        """Bridge an SDK ``victor_contracts.tools.ToolContract`` into internal metadata.

        FEP-0009 Phase 2. **Duck-typed by design** — this does NOT import
        ``victor_contracts`` (victor-ai's contracts floor is below the version that ships
        the contract), so it reads the contract's declared trait *values* off attributes.
        It starts from the autogen baseline for the fields the contract does not carry
        (keywords/use_cases/examples/priority_hints) and overrides with the declared
        traits; declared selection hints replace autogen only when non-empty.

        Args:
            contract: An object exposing ``category`` / ``access_mode`` / ``danger_level``
                / ``execution_category`` / ``cost_tier`` (str or str-enum) and optional
                ``keywords`` / ``use_cases`` / ``task_types`` / ``stages`` sequences.
            tool: The owning tool (for the autogen baseline: name/description/parameters).

        Returns:
            A :class:`ToolMetadata` fusing the declared contract over the autogen baseline.
        """

        def _v(value: Any) -> Any:
            return value.value if hasattr(value, "value") else value

        def _enum(enum_cls: Any, value: Any, default: Any) -> Any:
            if value is None:
                return default
            try:
                return enum_cls(value)
            except ValueError:
                return default

        cost_tier = _enum(CostTier, _v(getattr(contract, "cost_tier", None)), None)

        base = cls.generate_from_tool(
            name=getattr(tool, "name", "") or "",
            description=getattr(tool, "description", "") or "",
            parameters=getattr(tool, "parameters", {}) or {},
            cost_tier=cost_tier,
        )

        declared_category = _v(getattr(contract, "category", None))
        if declared_category:
            base.category = declared_category

        base.access_mode = _enum(
            AccessMode, _v(getattr(contract, "access_mode", None)), base.access_mode
        )
        base.danger_level = _enum(
            DangerLevel, _v(getattr(contract, "danger_level", None)), base.danger_level
        )
        base.execution_category = _enum(
            ExecutionCategory,
            _v(getattr(contract, "execution_category", None)),
            base.execution_category,
        )

        # Declared selection hints override the autogen baseline only when non-empty.
        for field_name in ("keywords", "use_cases", "task_types", "stages"):
            declared = getattr(contract, field_name, None)
            if declared:
                setattr(base, field_name, list(declared))

        # Per-argument kinds (FEP-0024): contract carries a tuple of (name, kind) pairs.
        declared_kinds = getattr(contract, "argument_kinds", None)
        if declared_kinds:
            base.argument_kinds = dict(declared_kinds)

        return base

    @classmethod
    def generate_from_tool(
        cls,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        cost_tier: Optional["CostTier"] = None,
    ) -> "ToolMetadata":
        """Auto-generate metadata from tool properties.

        This factory method creates sensible default metadata by extracting
        semantic information from the tool's name, description, and parameters.
        Tools that want more precise control should override the metadata property.

        Args:
            name: Tool name (e.g., 'git', 'file_search')
            description: Tool description text
            parameters: Tool parameters JSON schema
            cost_tier: Optional cost tier for priority hints

        Returns:
            ToolMetadata with auto-generated values
        """
        # Extract category from tool name (first part before underscore)
        name_parts = name.lower().replace("-", "_").split("_")
        category = name_parts[0] if name_parts else "general"

        # Known category mappings for common patterns
        category_mappings = {
            "git": "git",
            "file": "filesystem",
            "filesystem": "filesystem",
            "code": "code",
            "web": "web",
            "docker": "docker",
            "database": "database",
            "db": "database",
            "test": "testing",
            "lint": "code_quality",
            "format": "code_quality",
            "security": "security",
            "mcp": "mcp",
            "lsp": "lsp",
            "refactor": "refactoring",
            "search": "search",
            "semantic": "search",
            "pipeline": "pipeline",
            "ci": "pipeline",
            "coverage": "pipeline",
            "audit": "audit",
            "compliance": "audit",
            "merge": "merge",
            "conflict": "merge",
            "iac": "security",
            "terraform": "security",
            "kubernetes": "security",
        }

        # Try to find a matching category
        for part in name_parts:
            if part in category_mappings:
                category = category_mappings[part]
                break

        # Extract keywords from name and description
        keywords = cls._extract_keywords(name, description)

        # Generate use cases from description
        use_cases = cls._generate_use_cases(name, description)

        # Generate examples from name and parameters
        examples = cls._generate_examples(name, parameters)

        # Generate priority hints from cost tier
        priority_hints = []
        if cost_tier:
            tier_hints = {
                CostTier.FREE: ["Preferred for local operations", "No external costs"],
                CostTier.LOW: ["Efficient compute-only operation"],
                CostTier.MEDIUM: ["Makes external API calls"],
                CostTier.HIGH: ["Resource-intensive operation", "Use sparingly"],
            }
            priority_hints = tier_hints.get(cost_tier, [])

        return cls(
            category=category,
            keywords=keywords,
            use_cases=use_cases,
            examples=examples,
            priority_hints=priority_hints,
        )

    @staticmethod
    def _extract_keywords(name: str, description: str) -> List[str]:
        """Extract keywords from tool name and description.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            List of extracted keywords
        """
        keywords = set()

        # Add name variations
        keywords.add(name.lower())
        keywords.add(name.lower().replace("_", " "))

        # Split camelCase and snake_case
        name_words = re.findall(r"[a-z]+", name.lower())
        keywords.update(name_words)

        # Extract significant words from description (skip common words)
        stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "tool",
            "tools",
            "use",
            "using",
        }

        # Extract words from first sentence of description
        first_sentence = description.split(".")[0] if description else ""
        desc_words = re.findall(r"\b[a-z]{3,}\b", first_sentence.lower())
        significant_words = [w for w in desc_words if w not in stopwords]
        keywords.update(significant_words[:5])  # Limit to top 5

        return list(keywords)[:10]  # Cap at 10 keywords

    @staticmethod
    def _generate_use_cases(name: str, description: str) -> List[str]:
        """Generate use cases from description.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            List of use cases
        """
        use_cases = []

        # Primary use case from name
        name_readable = name.replace("_", " ").replace("-", " ")
        use_cases.append(f"{name_readable} operations")

        # Extract action verbs from description
        if description:
            # Look for common action patterns
            action_patterns = [
                (r"\b(create|generate|make)\b", "creating"),
                (r"\b(read|view|show|display|get)\b", "viewing"),
                (r"\b(update|modify|edit|change)\b", "modifying"),
                (r"\b(delete|remove|clear)\b", "removing"),
                (r"\b(search|find|locate|query)\b", "searching"),
                (r"\b(analyze|inspect|check|validate)\b", "analyzing"),
                (r"\b(execute|run|perform)\b", "executing"),
                (r"\b(list|enumerate)\b", "listing"),
            ]

            for pattern, action in action_patterns:
                if re.search(pattern, description.lower()):
                    use_cases.append(f"{action} {name_readable}")
                    break

        return use_cases[:3]  # Limit to 3 use cases

    @staticmethod
    def _generate_examples(name: str, parameters: Dict[str, Any]) -> List[str]:
        """Generate example requests from tool name and parameters.

        Args:
            name: Tool name
            parameters: Tool parameters JSON schema

        Returns:
            List of example requests
        """
        examples = []
        name_readable = name.replace("_", " ").replace("-", " ")

        # Basic example
        examples.append(f"use {name_readable}")

        # Parameter-based examples
        props = parameters.get("properties", {})
        if props:
            # Get first required parameter for more specific example
            required = parameters.get("required", [])
            if required:
                first_param = required[0]
                param_desc = props.get(first_param, {}).get("description", "")
                if param_desc:
                    examples.append(f"{name_readable} with {first_param}")

        return examples[:2]  # Limit to 2 examples


def _canonical_registry_cls():
    """Resolve the canonical metadata registry lazily to avoid import cycles."""
    from victor.tools.metadata_registry import (
        ToolMetadataRegistry as CanonicalToolMetadataRegistry,
    )

    return CanonicalToolMetadataRegistry


def _entry_to_tool_metadata(entry: Any) -> Optional[ToolMetadata]:
    """Convert a canonical ToolMetadataEntry into the legacy ToolMetadata shape."""
    if entry is None:
        return None

    return ToolMetadata(
        category=entry.category or "",
        keywords=sorted(entry.keywords),
        use_cases=[],
        examples=[],
        priority_hints=[],
        priority=entry.priority,
        access_mode=entry.access_mode,
        danger_level=entry.danger_level,
        stages=sorted(entry.stages),
        mandatory_keywords=sorted(entry.mandatory_keywords),
        task_types=sorted(entry.task_types),
        signature_params=sorted(entry.signature_params),
        execution_category=entry.execution_category,
    )


def _warn_legacy_registry_usage(stacklevel: int = 2) -> None:
    """Emit a one-time deprecation warning for the legacy registry wrapper."""
    global _LEGACY_METADATA_REGISTRY_WARNING_EMITTED
    if _LEGACY_METADATA_REGISTRY_WARNING_EMITTED:
        return

    warnings.warn(
        "Importing ToolMetadataRegistry from victor.tools.metadata is deprecated. "
        "Use 'from victor.tools.metadata_registry import ToolMetadataRegistry' instead. "
        "This compatibility wrapper will be removed in version 0.10.0.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    _LEGACY_METADATA_REGISTRY_WARNING_EMITTED = True


class ToolMetadataRegistry:
    """Compatibility wrapper over victor.tools.metadata_registry.ToolMetadataRegistry.

    This preserves the older API surface exported through ``victor.tools.base``
    while delegating all runtime state to the canonical registry implementation.
    """

    _instance: Optional["ToolMetadataRegistry"] = None

    def __init__(self, delegate: Optional[Any] = None) -> None:
        _warn_legacy_registry_usage(stacklevel=3)
        self._delegate = delegate or _canonical_registry_cls().get_instance()

    @classmethod
    def get_instance(cls) -> "ToolMetadataRegistry":
        """Get the singleton compatibility wrapper."""
        _warn_legacy_registry_usage(stacklevel=3)
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._delegate = _canonical_registry_cls().get_instance()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset both the compatibility wrapper and canonical singleton."""
        _warn_legacy_registry_usage(stacklevel=3)
        cls._instance = None
        _canonical_registry_cls().reset_instance()

    @staticmethod
    def _calculate_tools_hash(tools: List["BaseTool"]) -> str:
        """Delegate hash calculation to the canonical registry."""
        _warn_legacy_registry_usage(stacklevel=3)
        return _canonical_registry_cls()._calculate_tools_hash(tools)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def register_tool(self, tool: "BaseTool") -> None:
        self._delegate.register_tool(tool)

    def unregister_tool(self, tool_name: str) -> None:
        self._delegate.unregister_tool(tool_name)

    def get_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        return _entry_to_tool_metadata(self._delegate.get_metadata(tool_name))

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        return {
            name: metadata
            for name, metadata in (
                (name, _entry_to_tool_metadata(entry))
                for name, entry in self._delegate.get_all().items()
            )
            if metadata is not None
        }

    def get_tools_by_category(self, category: str) -> List[str]:
        return sorted(self._delegate.get_tools_by_category(category))

    def get_tools_by_access_mode(self, access_mode: "AccessMode") -> List[str]:
        """Return tool names whose metadata declares the given access mode.

        Single source of truth for "which tools modify files/state" — replaces
        hardcoded constants (e.g. the benchmark adapter's _FILE_MODIFYING_TOOLS)
        so UI approval prompts, security policies, and eval file-tracking all
        read the same core metadata. Falls back to AccessMode.READONLY when an
        entry has no access_mode set (the ToolMetadata default).
        """
        return sorted(
            name
            for name, meta in self.get_all_metadata().items()
            if (meta.access_mode or AccessMode.READONLY) == access_mode
        )

    def get_tools_by_keyword(self, keyword: str) -> List[str]:
        return sorted(entry.name for entry in self._delegate.get_by_keyword(keyword))

    def get_all_categories(self) -> List[str]:
        return sorted(self._delegate.get_all_categories())

    def get_all_keywords(self) -> List[str]:
        return sorted(self._delegate.get_all_keywords())

    def search_tools(self, query: str) -> List[str]:
        query_lower = query.lower()
        matches = set()

        for tool_name in self._delegate.get_all_tool_names():
            if query_lower in tool_name.lower():
                matches.add(tool_name)

        matches.update(self._delegate.get_tools_matching_text(query))

        for category in self._delegate.get_all_categories():
            if query_lower in category.lower():
                matches.update(self._delegate.get_tools_by_category(category))

        return sorted(matches)

    def export_all(self) -> Dict[str, Dict[str, Any]]:
        return {name: metadata.to_dict() for name, metadata in self.get_all_metadata().items()}

    def get_statistics(self) -> Dict[str, Any]:
        return self._delegate.get_statistics()

    def get_category_tools_map(self) -> Dict[str, List[str]]:
        return {
            category: sorted(self._delegate.get_tools_by_category(category))
            for category in self._delegate.get_all_categories()
        }

    def get_tools_for_task_type(self, task_type: str) -> List[str]:
        matched = self._delegate.get_tools_by_task_type(task_type)
        if matched:
            return sorted(matched)

        task_category_mapping = {
            "edit": ["filesystem", "code", "refactoring", "git"],
            "search": ["search", "code", "code_intel"],
            "analyze": ["code", "pipeline", "security", "audit", "code_quality"],
            "design": ["filesystem", "generation", "code"],
            "create": ["filesystem", "generation", "code", "testing"],
            "general": ["filesystem", "code", "search"],
        }

        categories = task_category_mapping.get(task_type, ["filesystem", "code"])
        tools = set()
        for category in categories:
            tools.update(self._delegate.get_tools_by_category(category))
        return sorted(tools)
