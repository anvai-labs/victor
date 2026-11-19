"""Canonical SDK-owned tool identifiers and alias helpers.

This module is intentionally dependency-free so vertical definition layers can
refer to stable tool identifiers without importing victor-ai runtime modules.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set


@dataclass(frozen=True)
class ToolNameEntry:
    """Immutable entry for a tool name with its aliases."""

    canonical: str
    aliases: FrozenSet[str] = field(default_factory=frozenset)
    deprecated: FrozenSet[str] = field(default_factory=frozenset)

    def all_names(self) -> Set[str]:
        """Return canonical plus alias names."""
        return {self.canonical} | set(self.aliases)


class ToolNames:
    """Registry of canonical tool names."""

    # Core Primitives (Cross-domain)
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    LS = "ls"
    PLAN = "plan"
    SHELL = "shell"
    SANDBOX = "sandbox"
    SANDBOX_UPLOAD = "sandbox_upload"
    GREP = "grep"
    SEARCH = "search"
    CODE_SEARCH = "code_search"
    WEB_SEARCH = "web_search"
    WEB_FETCH = "web_fetch"
    PATCH = "patch"
    DIFF = "diff"
    TEST = "test"
    HTTP = "http"
    MCP = "mcp"
    WORKFLOW = "workflow"
    OVERVIEW = "overview"
    SCAFFOLD = "scaffold"
    MEMORY_SET = "memory_set"
    MEMORY_FORGET = "memory_forget"

    # Unified command-shell domains (bash-style `domain action args`).
    # These are the primary LLM-facing tools; granular primitives above are
    # the implementations behind them (schema-hidden but still callable).
    FS = "fs"
    CODE = "code"
    WEB = "web"

    # Coding Vertical Specific (To be migrated)
    SYMBOL = "symbol"
    REFS = "refs"
    LSP = "lsp"
    RENAME = "rename"
    EXTRACT = "extract"
    INLINE = "inline"
    IMPORTS = "imports"
    GIT = "git"
    PR = "pr"
    COMMIT_MSG = "commit_msg"
    CONFLICTS = "conflicts"
    REVIEW = "review"
    METRICS = "metrics"
    SCAN = "scan"
    DEPS = "deps"

    # Research/Analysis Specific (To be migrated)
    SUMMARIZE = "summarize"
    DOCS = "docs"
    DOCS_COVERAGE = "docs_coverage"
    GRAPH = "graph"
    ARCH_SUMMARY = "arch_summary"

    # DevOps/Infrastructure Specific (To be migrated)
    DB = "db"
    DOCKER = "docker"
    CICD = "cicd"
    API_TEST = "api_test"
    BATCH = "batch"
    CACHE = "cache"
    MERGE = "merge"
    PIPELINE = "pipeline"
    AUDIT = "audit"
    IAC = "iac"

    @classmethod
    def file_operations(cls) -> tuple[str, str, str, str]:
        """Return the canonical file-operation tool set shared across verticals."""

        return (cls.READ, cls.WRITE, cls.EDIT, cls.GREP)


TOOL_ALIASES: Dict[str, str] = {
    "read_file": ToolNames.READ,
    "write_file": ToolNames.WRITE,
    "edit_files": ToolNames.EDIT,
    "edit_file": ToolNames.EDIT,
    "patch_file": ToolNames.EDIT,
    "list_directory": ToolNames.LS,
    "plan_files": ToolNames.PLAN,
    "get_project_overview": ToolNames.OVERVIEW,
    "execute_bash": ToolNames.SHELL,
    "run": ToolNames.SHELL,
    "bash": ToolNames.SHELL,
    "execute": ToolNames.SHELL,
    "cmd": ToolNames.SHELL,
    "container.exec": ToolNames.SHELL,
    "container.execute": ToolNames.SHELL,
    "container.shell": ToolNames.SHELL,
    "container.run": ToolNames.SHELL,
    "container.command": ToolNames.SHELL,
    "container.file.read": ToolNames.READ,
    "container.file.write": ToolNames.WRITE,
    "container.file.append": ToolNames.WRITE,
    "container.file.list": ToolNames.LS,
    "container.file.delete": ToolNames.WRITE,
    "repo_browser.read_file": ToolNames.READ,
    "repo_browser.write_file": ToolNames.WRITE,
    "repo_browser.list_directory": ToolNames.LS,
    "tool.read_file": ToolNames.READ,
    "tool.write_file": ToolNames.WRITE,
    "tool.list_directory": ToolNames.LS,
    "execute_python_in_sandbox": ToolNames.SANDBOX,
    "upload_files_to_sandbox": ToolNames.SANDBOX_UPLOAD,
    # Unified search tool (grep, files subcommands)
    "search": ToolNames.SEARCH,
    # Legacy code_search aliases (kept for backward compatibility)
    "code_search": ToolNames.CODE_SEARCH,
    "semantic_code_search": ToolNames.CODE_SEARCH,
    "web_summarize": ToolNames.SUMMARIZE,
    "find_symbol": ToolNames.SYMBOL,
    "find_references": ToolNames.REFS,
    "rename_symbol": ToolNames.RENAME,
    "architecture_summary": ToolNames.ARCH_SUMMARY,
    "refactor_rename_symbol": ToolNames.RENAME,
    "refactor_extract_function": ToolNames.EXTRACT,
    "refactor_inline_variable": ToolNames.INLINE,
    "refactor_organize_imports": ToolNames.IMPORTS,
    "git_status": ToolNames.GIT,
    "git_diff": ToolNames.GIT,
    "git_log": ToolNames.GIT,
    "git_commit": ToolNames.GIT,
    "git_branch": ToolNames.GIT,
    "git_stage": ToolNames.GIT,
    "git_create_pr": ToolNames.PR,
    "git_suggest_commit": ToolNames.COMMIT_MSG,
    "git_analyze_conflicts": ToolNames.CONFLICTS,
    "apply_patch": ToolNames.PATCH,
    "create_patch": ToolNames.DIFF,
    "generate_docs": ToolNames.DOCS,
    "analyze_docs": ToolNames.DOCS_COVERAGE,
    "code_review": ToolNames.REVIEW,
    "analyze_metrics": ToolNames.METRICS,
    "security_scan": ToolNames.SCAN,
    "run_tests": ToolNames.TEST,
    "database": ToolNames.DB,
    "dependency": ToolNames.DEPS,
    "http_request": ToolNames.HTTP,
    "mcp_call": ToolNames.MCP,
    "run_workflow": ToolNames.WORKFLOW,
    "merge_conflicts": ToolNames.MERGE,
    "pipeline_analyzer": ToolNames.PIPELINE,
    "iac_scanner": ToolNames.IAC,
}

CANONICAL_TO_ALIASES: Dict[str, Set[str]] = {}
for alias, canonical in TOOL_ALIASES.items():
    CANONICAL_TO_ALIASES.setdefault(canonical, set()).add(alias)


def get_canonical_name(name: str) -> str:
    """Resolve a tool name to its canonical form."""

    return TOOL_ALIASES.get(name, name)


def get_aliases(canonical_name: str) -> Set[str]:
    """Get all aliases for a canonical tool name."""

    return CANONICAL_TO_ALIASES.get(canonical_name, set())


def is_valid_tool_name(name: str) -> bool:
    """Check whether a name is a canonical tool name or a supported alias."""

    if name in TOOL_ALIASES:
        return True
    return name in CANONICAL_TO_ALIASES or hasattr(ToolNames, name.upper())


def get_all_canonical_names() -> Set[str]:
    """Get all canonical tool names."""

    return {
        getattr(ToolNames, attr)
        for attr in dir(ToolNames)
        if not attr.startswith("_") and isinstance(getattr(ToolNames, attr), str)
    }


def get_name_mapping() -> Dict[str, str]:
    """Get full mapping of canonical names and aliases to canonical names."""

    mapping = {canonical: canonical for canonical in get_all_canonical_names()}
    mapping.update(TOOL_ALIASES)
    return mapping


__all__ = [
    "ToolNames",
    "ToolNameEntry",
    "TOOL_ALIASES",
    "CANONICAL_TO_ALIASES",
    "get_canonical_name",
    "get_aliases",
    "is_valid_tool_name",
    "get_all_canonical_names",
    "get_name_mapping",
]
