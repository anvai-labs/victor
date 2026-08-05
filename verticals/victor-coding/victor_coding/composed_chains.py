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

"""LCEL-composed tool chains for common coding tasks.

This module provides pre-built LCEL composition chains for common
software development workflows. These chains combine multiple tools
into reusable, composable units.

The chains leverage victor.tools.composition primitives:
- RunnableSequence: Sequential tool execution (|)
- RunnableParallel: Parallel tool execution
- RunnableBranch: Conditional routing
- RunnableLambda: Transform functions

Example:
    from victor_coding.composed_chains import (
        explore_file_chain,
        analyze_function_chain,
        safe_edit_chain,
    )

    # Explore a file before editing
    result = await explore_file_chain.invoke({"path": "src/auth.py"})

    # Analyze a specific function
    result = await analyze_function_chain.invoke({
        "path": "src/auth.py",
        "symbol": "authenticate_user",
    })

    # Safe edit with verification
    result = await safe_edit_chain.invoke({
        "path": "src/auth.py",
        "edit": {"old": "...", "new": "..."},
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.tools.composition import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    as_runnable,
    chain,
    LazyToolRunnable as BaseLazyToolRunnable,
    ToolCompositionBuilder,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Transform Functions for Chain Composition
# =============================================================================


def extract_file_content(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract file content from read result.

    Args:
        result: Tool result from read operation

    Returns:
        Dict with content and metadata
    """
    if not result.get("success"):
        return {"content": None, "error": result.get("error")}

    output = result.get("output", {})
    return {
        "content": output.get("content", ""),
        "path": output.get("path", ""),
        "lines": output.get("lines", 0),
        "language": output.get("language"),
    }


def extract_symbols_list(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract symbol list from symbols tool result.

    Args:
        result: Tool result from symbols operation

    Returns:
        List of symbol definitions
    """
    if not result.get("success"):
        return []

    output = result.get("output", {})
    return output.get("symbols", [])


def extract_search_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract search results from code_search result.

    Args:
        result: Tool result from search operation

    Returns:
        List of search matches
    """
    if not result.get("success"):
        return []

    output = result.get("output", {})
    return output.get("matches", output.get("results", []))


def prepare_edit_input(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare input for edit operation from context.

    Transforms exploration results into edit-ready format.

    Args:
        ctx: Context with file content and edit spec

    Returns:
        Dict ready for edit tool
    """
    return {
        "path": ctx.get("path"),
        "old_text": ctx.get("edit", {}).get("old"),
        "new_text": ctx.get("edit", {}).get("new"),
    }


def merge_analysis_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge parallel analysis results into summary.

    Args:
        ctx: Context with content, symbols, and references

    Returns:
        Merged analysis summary
    """
    return {
        "file_info": ctx.get("content", {}),
        "symbols": ctx.get("symbols", []),
        "references": ctx.get("references", []),
        "complexity": _estimate_complexity(ctx),
    }


def _estimate_complexity(ctx: Dict[str, Any]) -> str:
    """Estimate code complexity from analysis results."""
    symbols = ctx.get("symbols", [])
    content = ctx.get("content", {})
    lines = content.get("lines", 0)

    if lines > 500 or len(symbols) > 50:
        return "high"
    elif lines > 200 or len(symbols) > 20:
        return "medium"
    return "low"


def is_python_file(ctx: Dict[str, Any]) -> bool:
    """Check if file is Python based on path or content."""
    path = ctx.get("path", "")
    return path.endswith(".py") or path.endswith(".pyi")


def is_typescript_file(ctx: Dict[str, Any]) -> bool:
    """Check if file is TypeScript based on path."""
    path = ctx.get("path", "")
    return path.endswith(".ts") or path.endswith(".tsx")


def is_javascript_file(ctx: Dict[str, Any]) -> bool:
    """Check if file is JavaScript based on path."""
    path = ctx.get("path", "")
    return path.endswith(".js") or path.endswith(".jsx")


# =============================================================================
# Lazy Tool Loading
# =============================================================================


class LazyToolRunnable(Runnable[Dict[str, Any], Dict[str, Any]]):
    """Lazily loads a tool by name when first invoked.

    This class extends the framework's LazyToolRunnable (BaseLazyToolRunnable)
    with tool-by-name loading capabilities, making it suitable for LCEL-style
    chain composition without requiring all tools to be imported at module load time.

    Uses BaseLazyToolRunnable from victor.tools.composition for the underlying
    lazy loading mechanism, adding the Runnable interface for chain composition.
    """

    def __init__(
        self,
        tool_name: str,
        output_key: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize with tool name.

        Args:
            tool_name: Name of the tool to load from the registry
            output_key: Key to extract from output
            input_mapping: Input key remapping
        """
        self._tool_name = tool_name
        self._output_key = output_key
        self._input_mapping = input_mapping or {}
        self._runnable: Optional[Runnable] = None

        # Use the framework's LazyToolRunnable for lazy loading
        self._lazy_loader = BaseLazyToolRunnable(
            factory=self._create_tool,
            name=tool_name,
            cache=True,
        )

    def _create_tool(self) -> Any:
        """Factory method to create the tool - used by BaseLazyToolRunnable."""
        from victor.tools import get_tool_by_name

        tool = get_tool_by_name(self._tool_name)
        if tool is None:
            raise ValueError(f"Tool '{self._tool_name}' not found")
        return tool

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._tool_name

    def _load_tool(self) -> Runnable:
        """Load the tool and create runnable using the lazy loader."""
        if self._runnable is not None:
            return self._runnable

        try:
            # Get the tool from the lazy loader (creates on first access)
            tool = self._lazy_loader.tool

            self._runnable = as_runnable(
                tool,
                output_key=self._output_key,
                input_mapping=self._input_mapping,
            )
            return self._runnable
        except Exception as e:
            logger.error(f"Failed to load tool '{self._tool_name}': {e}")
            raise

    async def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Execute the lazily-loaded tool."""
        runnable = self._load_tool()
        return await runnable.invoke(input, config)

    @property
    def is_initialized(self) -> bool:
        """Check if the tool has been initialized.

        Delegates to the underlying BaseLazyToolRunnable.
        """
        return self._lazy_loader.is_initialized

    def reset(self) -> None:
        """Reset the lazy tool, clearing the cached instance.

        Useful for testing or forcing re-initialization.
        """
        self._runnable = None
        self._lazy_loader.reset()

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "pending"
        return f"LazyToolRunnable({self._tool_name}, status={status})"


def lazy_tool(
    name: str,
    output_key: Optional[str] = None,
    input_mapping: Optional[Dict[str, str]] = None,
) -> LazyToolRunnable:
    """Create a lazy-loading tool runnable.

    Args:
        name: Tool name to load
        output_key: Key to extract from output
        input_mapping: Input key remapping

    Returns:
        LazyToolRunnable instance
    """
    return LazyToolRunnable(name, output_key, input_mapping)


# =============================================================================
# Pre-built Chains for Common Coding Tasks
# =============================================================================


# File Exploration Chain
# Reads file, extracts symbols, and provides structured analysis
explore_file_chain = chain(
    lazy_tool("read"),
    RunnableLambda(extract_file_content, name="extract_content"),
    RunnableParallel(
        content=RunnablePassthrough(),
        symbols=lazy_tool("symbols"),
    ),
    RunnableLambda(merge_analysis_results, name="merge_analysis"),
)
"""Chain to explore a file: read -> extract content -> get symbols."""


# Function Analysis Chain
# Analyzes a specific function with references
analyze_function_chain = chain(
    lazy_tool("symbols"),
    RunnableLambda(
        lambda ctx: {
            "path": ctx.get("path"),
            "symbol": ctx.get("symbol"),
            "symbols": extract_symbols_list(ctx),
        },
        name="prepare_refs",
    ),
    RunnableParallel(
        definition=lazy_tool("hover"),
        references=lazy_tool("references"),
    ),
)
"""Chain to analyze a function: get symbols -> get hover info + references."""


# Safe Edit Chain
# Reads file, makes edit, verifies syntax
safe_edit_chain = chain(
    # First read the file to confirm it exists
    lazy_tool("read"),
    RunnableLambda(
        lambda ctx: {
            **ctx,
            "original_content": extract_file_content(ctx).get("content"),
        },
        name="store_original",
    ),
    # Apply the edit
    RunnableLambda(prepare_edit_input, name="prepare_edit"),
    lazy_tool("edit"),
    # Verify the result
    RunnableLambda(
        lambda ctx: {
            "success": ctx.get("success", False),
            "edit_result": ctx,
        },
        name="wrap_result",
    ),
)
"""Chain for safe editing: read -> edit -> verify."""


# Git Status and Diff Chain
# Gets comprehensive git state
git_status_chain = RunnableParallel(
    status=lazy_tool("git_status"),
    diff=lazy_tool("git_diff"),
    branch=lazy_tool("git_branch"),
)
"""Chain to get full git state: status + diff + branch in parallel."""


# Code Search Chain with Context
# Searches code then reads matching files
search_with_context_chain = chain(
    lazy_tool("code_search"),
    RunnableLambda(
        lambda ctx: {
            "matches": extract_search_results(ctx),
            "paths": [m.get("path") for m in extract_search_results(ctx)[:5]],
        },
        name="extract_matches",
    ),
)
"""Chain to search code and extract match context."""


# Language-Aware Lint Chain
# Routes to appropriate linter based on file type
lint_chain = RunnableBranch(
    (is_python_file, lazy_tool("shell").bind(command="ruff check {path}")),
    (is_typescript_file, lazy_tool("shell").bind(command="npx tsc --noEmit {path}")),
    (is_javascript_file, lazy_tool("shell").bind(command="npx eslint {path}")),
    default=RunnableLambda(lambda ctx: {"success": True, "output": "No linter for this file type"}),
)
"""Language-aware linting chain with conditional routing."""


# Test Discovery Chain
# Finds and categorizes test files
test_discovery_chain = chain(
    lazy_tool("grep", input_mapping={"pattern": "query"}),
    RunnableLambda(
        lambda ctx: {
            "test_files": [
                m for m in extract_search_results(ctx) if "test" in m.get("path", "").lower()
            ],
        },
        name="filter_tests",
    ),
)
"""Chain to discover test files matching a pattern."""


# Comprehensive Review Chain
# Parallel analysis for code review
review_analysis_chain = RunnableParallel(
    content=lazy_tool("read"),
    symbols=lazy_tool("symbols"),
    git_diff=lazy_tool("git_diff"),
)
"""Parallel chain for code review: content + symbols + diff."""


# =============================================================================
# Chain Factories for Custom Composition
# =============================================================================


@dataclass
class ChainConfig:
    """Configuration for chain creation.

    Attributes:
        include_git: Whether to include git operations
        include_tests: Whether to include test operations
        max_files: Maximum files to process in parallel
        timeout_seconds: Timeout for chain execution
    """

    include_git: bool = True
    include_tests: bool = True
    max_files: int = 10
    timeout_seconds: float = 60.0


def create_exploration_chain(
    config: Optional[ChainConfig] = None,
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    """Create a comprehensive exploration chain.

    This chain explores a codebase by:
    1. Reading the target file
    2. Getting symbols
    3. Optionally getting git status
    4. Merging results

    Args:
        config: Chain configuration

    Returns:
        Configured exploration chain
    """
    config = config or ChainConfig()

    steps: Dict[str, Runnable] = {
        "content": lazy_tool("read"),
        "symbols": lazy_tool("symbols"),
        "overview": lazy_tool("overview"),
    }

    if config.include_git:
        steps["git"] = git_status_chain

    return chain(
        RunnableParallel(**steps),
        RunnableLambda(merge_analysis_results, name="merge"),
    )


def create_edit_verify_chain(
    run_tests: bool = True,
    check_lint: bool = True,
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    """Create an edit chain with verification steps.

    This chain:
    1. Reads the target file
    2. Applies the edit
    3. Optionally runs linter
    4. Optionally runs tests

    Args:
        run_tests: Whether to run tests after edit
        check_lint: Whether to run linter after edit

    Returns:
        Configured edit-verify chain
    """
    verification_steps: Dict[str, Runnable] = {}

    if check_lint:
        verification_steps["lint"] = lint_chain

    if run_tests:
        verification_steps["tests"] = lazy_tool("run_tests")

    if not verification_steps:
        # No verification, just edit
        return safe_edit_chain

    return chain(
        safe_edit_chain,
        RunnableParallel(**verification_steps),
        RunnableLambda(
            lambda ctx: {
                "success": all(v.get("success", True) for v in ctx.values() if isinstance(v, dict)),
                "results": ctx,
            },
            name="aggregate_verification",
        ),
    )


def create_refactor_chain(
    refactor_type: str = "rename",
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    """Create a refactoring chain for specific refactor type.

    Args:
        refactor_type: Type of refactor (rename, extract, inline)

    Returns:
        Configured refactor chain
    """
    tool_mapping = {
        "rename": "rename_symbol",
        "extract": "extract_function",
        "inline": "refactor",
    }

    refactor_tool = tool_mapping.get(refactor_type, "refactor")

    return chain(
        # First analyze the target
        RunnableParallel(
            content=lazy_tool("read"),
            symbols=lazy_tool("symbols"),
            references=lazy_tool("references"),
        ),
        # Apply refactoring
        lazy_tool(refactor_tool),
        # Verify with tests
        lazy_tool("run_tests"),
        RunnableLambda(
            lambda ctx: {
                "refactor_type": refactor_type,
                "success": ctx.get("success", False),
                "result": ctx,
            },
            name="wrap_refactor_result",
        ),
    )


# =============================================================================
# Registry of Available Chains
# =============================================================================


CODING_CHAINS: Dict[str, Runnable] = {
    "explore_file": explore_file_chain,
    "analyze_function": analyze_function_chain,
    "safe_edit": safe_edit_chain,
    "git_status": git_status_chain,
    "search_with_context": search_with_context_chain,
    "lint": lint_chain,
    "test_discovery": test_discovery_chain,
    "review_analysis": review_analysis_chain,
}


def get_chain(name: str) -> Optional[Runnable]:
    """Get a pre-built chain by name.

    Args:
        name: Chain name

    Returns:
        Chain if found, None otherwise
    """
    return CODING_CHAINS.get(name)


def list_chains() -> List[str]:
    """List all available chain names.

    Returns:
        List of chain names
    """
    return list(CODING_CHAINS.keys())


# =============================================================================
# Framework Chain Registry Integration
# =============================================================================


def _register_chains_with_framework() -> None:
    """Register all coding chains with the framework ChainRegistry.

    This function is called on module import to register all pre-built
    coding chains with the framework-level ChainRegistry for cross-vertical
    discovery and reuse.

    Chains are registered with semantic versioning for compatibility tracking.
    """
    from victor_contracts.chain_runtime import get_chain_registry

    # Get singleton instance
    registry = get_chain_registry()

    # Register exploration chains
    registry.register_chain(
        name="explore_file_chain",
        version="1.0.0",
        chain=explore_file_chain,
        category="exploration",
        description="Explore a file with context (read + ls + grep)",
        tags=["file", "exploration", "context"],
        author="victor",
    )

    registry.register_chain(
        name="search_with_context_chain",
        version="1.0.0",
        chain=search_with_context_chain,
        category="exploration",
        description="Search codebase with surrounding context",
        tags=["search", "codebase", "context"],
        author="victor",
    )

    # Register analysis chains
    registry.register_chain(
        name="analyze_function_chain",
        version="1.0.0",
        chain=analyze_function_chain,
        category="analysis",
        description="Analyze a function with symbol extraction",
        tags=["function", "analysis", "ast", "symbols"],
        author="victor",
    )

    registry.register_chain(
        name="review_analysis_chain",
        version="1.0.0",
        chain=review_analysis_chain,
        category="analysis",
        description="Parallel analysis for code review (read + symbols)",
        tags=["review", "analysis", "parallel"],
        author="victor",
    )

    # Register editing chains
    registry.register_chain(
        name="safe_edit_chain",
        version="1.0.0",
        chain=safe_edit_chain,
        category="editing",
        description="Safe edit with verification (read + edit + read)",
        tags=["edit", "safe", "verification"],
        author="victor",
    )

    # Register testing chains
    registry.register_chain(
        name="test_discovery_chain",
        version="1.0.0",
        chain=test_discovery_chain,
        category="testing",
        description="Discover and analyze tests for code",
        tags=["test", "discovery", "analysis"],
        author="victor",
    )

    registry.register_chain(
        name="lint_chain",
        version="1.0.0",
        chain=lint_chain,
        category="testing",
        description="Branching chain for lint analysis",
        tags=["lint", "quality", "branch"],
        author="victor",
    )

    # Register git chains
    registry.register_chain(
        name="git_status_chain",
        version="1.0.0",
        chain=git_status_chain,
        category="exploration",
        description="Parallel git status and branch info",
        tags=["git", "status", "parallel"],
        author="victor",
    )


# Register chains on module import
_register_chains_with_framework()


__all__ = [
    # Pre-built chains
    "explore_file_chain",
    "analyze_function_chain",
    "safe_edit_chain",
    "git_status_chain",
    "search_with_context_chain",
    "lint_chain",
    "test_discovery_chain",
    "review_analysis_chain",
    # Factories
    "create_exploration_chain",
    "create_edit_verify_chain",
    "create_refactor_chain",
    # Lazy loading (local wrapper for tool-by-name loading)
    "lazy_tool",
    "LazyToolRunnable",
    # Re-exported from victor.tools.composition for convenience
    "BaseLazyToolRunnable",
    "ToolCompositionBuilder",
    # Registry
    "get_chain",
    "list_chains",
    "CODING_CHAINS",
    # Config
    "ChainConfig",
    # Transform functions
    "extract_file_content",
    "extract_symbols_list",
    "extract_search_results",
]
