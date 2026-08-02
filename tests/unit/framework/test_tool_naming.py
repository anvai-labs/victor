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

"""Tests for victor.framework.tool_naming module.

This module tests the framework-level tool naming utilities including:
- canonicalize_tool_set()
- canonicalize_tool_dict()
- canonicalize_tool_list()
- canonicalize_transitions()
- canonicalize_dependencies()
- validate_tool_names()
- get_legacy_names_report()
- Re-exported utilities from victor.tools.tool_names
"""

import logging
import pytest

from victor.framework.tool_naming import (
    CANONICAL_TO_ALIASES,
    TOOL_ALIASES,
    ToolNameEntry,
    ToolNames,
    canonicalize_dependencies,
    canonicalize_tool_dict,
    canonicalize_tool_list,
    canonicalize_tool_set,
    canonicalize_transitions,
    get_aliases,
    get_all_canonical_names,
    get_canonical_name,
    get_legacy_names_report,
    get_name_mapping,
    is_valid_tool_name,
    validate_tool_names,
)
from victor.core.tool_types import ToolDependency


class TestToolNamesConstants:
    """Tests for ToolNames class constants."""

    def test_core_filesystem_tools(self):
        """Verify core filesystem tool names are defined."""
        assert ToolNames.READ == "read"
        assert ToolNames.WRITE == "write"
        assert ToolNames.EDIT == "edit"
        assert ToolNames.LS == "ls"
        assert ToolNames.PLAN == "plan"

    def test_shell_tools(self):
        """Verify shell/command execution tool names."""
        assert ToolNames.SHELL == "shell"
        assert not hasattr(ToolNames, "SHELL_READONLY")
        assert ToolNames.SANDBOX == "sandbox"
        assert ToolNames.SANDBOX_UPLOAD == "sandbox_upload"

    def test_search_tools(self):
        """Verify search tool names."""
        assert ToolNames.GREP == "grep"
        assert ToolNames.CODE_SEARCH == "code_search"
        assert ToolNames.WEB_SEARCH == "web_search"
        assert ToolNames.WEB_FETCH == "web_fetch"
        assert ToolNames.SUMMARIZE == "summarize"

    def test_code_intelligence_tools(self):
        """Verify code intelligence tool names."""
        assert ToolNames.SYMBOL == "symbol"
        assert ToolNames.REFS == "refs"
        assert ToolNames.LSP == "lsp"

    def test_refactoring_tools(self):
        """Verify refactoring tool names."""
        assert ToolNames.RENAME == "rename"
        assert ToolNames.EXTRACT == "extract"
        assert ToolNames.INLINE == "inline"
        assert ToolNames.IMPORTS == "imports"

    def test_git_tools(self):
        """Verify git tool names."""
        assert ToolNames.GIT == "git"
        assert ToolNames.PR == "pr"
        assert ToolNames.COMMIT_MSG == "commit_msg"
        assert ToolNames.CONFLICTS == "conflicts"

    def test_patch_diff_tools(self):
        """Verify patch/diff tool names."""
        assert ToolNames.PATCH == "patch"
        assert ToolNames.DIFF == "diff"

    def test_documentation_tools(self):
        """Verify documentation tool names."""
        assert ToolNames.DOCS == "docs"
        assert ToolNames.DOCS_COVERAGE == "docs_coverage"

    def test_code_quality_tools(self):
        """Verify code quality tool names."""
        assert ToolNames.REVIEW == "review"
        assert ToolNames.METRICS == "metrics"
        assert ToolNames.SCAN == "scan"
        assert ToolNames.TEST == "test"

    def test_infrastructure_tools(self):
        """Verify infrastructure tool names."""
        assert ToolNames.DB == "db"
        assert ToolNames.DOCKER == "docker"
        assert ToolNames.CICD == "cicd"
        assert ToolNames.DEPS == "deps"
        assert ToolNames.HTTP == "http"
        assert ToolNames.API_TEST == "api_test"

    def test_utility_tools(self):
        """Verify utility tool names."""
        assert ToolNames.BATCH == "batch"
        assert ToolNames.CACHE == "cache"
        assert ToolNames.MCP == "mcp"
        assert ToolNames.WORKFLOW == "workflow"
        assert ToolNames.SCAFFOLD == "scaffold"
        assert ToolNames.OVERVIEW == "overview"

    def test_analysis_tools(self):
        """Verify analysis tool names."""
        assert ToolNames.MERGE == "merge"
        assert ToolNames.PIPELINE == "pipeline"
        assert ToolNames.AUDIT == "audit"
        assert ToolNames.IAC == "iac"
        assert ToolNames.GRAPH == "graph"
        assert ToolNames.ARCH_SUMMARY == "arch_summary"


class TestToolNameEntry:
    """Tests for ToolNameEntry dataclass."""

    def test_create_basic_entry(self):
        """Create basic entry with just canonical name."""
        entry = ToolNameEntry(canonical="read")
        assert entry.canonical == "read"
        assert entry.aliases == frozenset()
        assert entry.deprecated == frozenset()

    def test_create_entry_with_aliases(self):
        """Create entry with aliases."""
        entry = ToolNameEntry(
            canonical="shell",
            aliases=frozenset({"execute_bash", "bash", "run"}),
        )
        assert entry.canonical == "shell"
        assert "execute_bash" in entry.aliases
        assert "bash" in entry.aliases
        assert "run" in entry.aliases

    def test_create_entry_with_deprecated(self):
        """Create entry with deprecated names."""
        entry = ToolNameEntry(
            canonical="shell",
            aliases=frozenset({"execute_bash"}),
            deprecated=frozenset({"old_shell"}),
        )
        assert "old_shell" in entry.deprecated

    def test_all_names_returns_canonical_and_aliases(self):
        """all_names() returns canonical name plus aliases."""
        entry = ToolNameEntry(
            canonical="read",
            aliases=frozenset({"read_file"}),
        )
        names = entry.all_names()
        assert "read" in names
        assert "read_file" in names
        assert len(names) == 2

    def test_all_names_only_canonical(self):
        """all_names() returns just canonical when no aliases."""
        entry = ToolNameEntry(canonical="git")
        names = entry.all_names()
        assert names == {"git"}

    def test_entry_is_immutable(self):
        """ToolNameEntry is frozen (immutable)."""
        entry = ToolNameEntry(canonical="read")
        with pytest.raises(AttributeError):
            entry.canonical = "write"


class TestToolAliases:
    """Tests for TOOL_ALIASES mapping."""

    def test_filesystem_aliases(self):
        """Verify filesystem tool aliases."""
        assert TOOL_ALIASES["read_file"] == "read"
        assert TOOL_ALIASES["write_file"] == "write"
        assert TOOL_ALIASES["edit_files"] == "edit"
        assert TOOL_ALIASES["edit_file"] == "edit"
        assert TOOL_ALIASES["list_directory"] == "ls"
        assert TOOL_ALIASES["plan_files"] == "plan"
        assert TOOL_ALIASES["get_project_overview"] == "overview"

    def test_shell_aliases(self):
        """Verify shell execution aliases."""
        assert TOOL_ALIASES["execute_bash"] == "shell"
        assert "shell_readonly" not in TOOL_ALIASES
        assert TOOL_ALIASES["run"] == "shell"
        assert TOOL_ALIASES["bash"] == "shell"
        assert TOOL_ALIASES["execute"] == "shell"
        assert TOOL_ALIASES["cmd"] == "shell"
        assert TOOL_ALIASES["execute_python_in_sandbox"] == "sandbox"
        assert TOOL_ALIASES["upload_files_to_sandbox"] == "sandbox_upload"

    def test_search_aliases(self):
        """Verify search tool aliases.

        ``search`` (the unified grep/files search tool) and ``code_search``
        are distinct canonical tools — ``search`` maps to itself, not to
        ``code_search``. ``semantic_code_search`` remains a legacy alias.
        """
        # code_search is canonical and maps to itself
        assert TOOL_ALIASES["code_search"] == "code_search"
        # search is its own canonical name (the unified search tool)
        assert TOOL_ALIASES["search"] == "search"
        # semantic_code_search is a legacy alias for code_search
        assert TOOL_ALIASES["semantic_code_search"] == "code_search"
        # web_summarize is an alias for summarize
        assert TOOL_ALIASES["web_summarize"] == "summarize"

    def test_code_intelligence_aliases(self):
        """Verify code intelligence aliases."""
        assert TOOL_ALIASES["find_symbol"] == "symbol"
        assert TOOL_ALIASES["find_references"] == "refs"
        assert TOOL_ALIASES["rename_symbol"] == "rename"
        assert TOOL_ALIASES["architecture_summary"] == "arch_summary"

    def test_refactoring_aliases(self):
        """Verify refactoring aliases."""
        assert TOOL_ALIASES["refactor_rename_symbol"] == "rename"
        assert TOOL_ALIASES["refactor_extract_function"] == "extract"
        assert TOOL_ALIASES["refactor_inline_variable"] == "inline"
        assert TOOL_ALIASES["refactor_organize_imports"] == "imports"

    def test_git_aliases(self):
        """Verify git operation aliases all map to 'git'."""
        git_aliases = [
            "git_status",
            "git_diff",
            "git_log",
            "git_commit",
            "git_branch",
            "git_stage",
        ]
        for alias in git_aliases:
            assert TOOL_ALIASES[alias] == "git"

    def test_git_specific_tool_aliases(self):
        """Verify PR and commit message tool aliases."""
        assert TOOL_ALIASES["git_create_pr"] == "pr"
        assert TOOL_ALIASES["git_suggest_commit"] == "commit_msg"
        assert TOOL_ALIASES["git_analyze_conflicts"] == "conflicts"

    def test_patch_aliases(self):
        """Verify patch/diff aliases."""
        assert TOOL_ALIASES["apply_patch"] == "patch"
        assert TOOL_ALIASES["create_patch"] == "diff"

    def test_documentation_aliases(self):
        """Verify documentation aliases."""
        assert TOOL_ALIASES["generate_docs"] == "docs"
        assert TOOL_ALIASES["analyze_docs"] == "docs_coverage"

    def test_code_quality_aliases(self):
        """Verify code quality aliases."""
        assert TOOL_ALIASES["code_review"] == "review"
        assert TOOL_ALIASES["analyze_metrics"] == "metrics"
        assert TOOL_ALIASES["security_scan"] == "scan"
        assert TOOL_ALIASES["run_tests"] == "test"

    def test_infrastructure_aliases(self):
        """Verify infrastructure aliases."""
        assert TOOL_ALIASES["database"] == "db"
        assert TOOL_ALIASES["dependency"] == "deps"
        assert TOOL_ALIASES["http_request"] == "http"
        assert TOOL_ALIASES["mcp_call"] == "mcp"
        assert TOOL_ALIASES["run_workflow"] == "workflow"

    def test_analysis_tool_aliases(self):
        """Verify analysis tool aliases."""
        assert TOOL_ALIASES["merge_conflicts"] == "merge"
        assert TOOL_ALIASES["pipeline_analyzer"] == "pipeline"
        assert TOOL_ALIASES["iac_scanner"] == "iac"


class TestCanonicalToAliases:
    """Tests for CANONICAL_TO_ALIASES reverse mapping."""

    def test_shell_has_many_aliases(self):
        """shell has multiple aliases."""
        aliases = CANONICAL_TO_ALIASES.get("shell", set())
        assert "execute_bash" in aliases
        assert "bash" in aliases
        assert "run" in aliases
        assert "execute" in aliases
        assert "cmd" in aliases

    def test_git_has_operation_aliases(self):
        """git has operation-specific aliases."""
        aliases = CANONICAL_TO_ALIASES.get("git", set())
        assert "git_status" in aliases
        assert "git_diff" in aliases
        assert "git_log" in aliases

    def test_canonical_without_aliases_not_in_map(self):
        """Canonical names without aliases may not be in reverse map."""
        # 'docker', 'cicd', 'batch', 'cache' have no aliases
        # They should not appear in CANONICAL_TO_ALIASES
        assert "docker" not in CANONICAL_TO_ALIASES
        assert "cicd" not in CANONICAL_TO_ALIASES


class TestGetCanonicalName:
    """Tests for get_canonical_name() function."""

    def test_alias_returns_canonical(self):
        """Aliases return their canonical form."""
        assert get_canonical_name("execute_bash") == "shell"
        assert get_canonical_name("shell_readonly") == "shell_readonly"
        assert get_canonical_name("read_file") == "read"
        assert get_canonical_name("edit_files") == "edit"
        assert get_canonical_name("edit_file") == "edit"
        assert get_canonical_name("run_tests") == "test"

    def test_canonical_returns_itself(self):
        """Canonical names return themselves."""
        assert get_canonical_name("shell") == "shell"
        assert get_canonical_name("read") == "read"
        assert get_canonical_name("edit") == "edit"
        assert get_canonical_name("git") == "git"

    def test_unknown_name_returns_itself(self):
        """Unknown names return themselves."""
        assert get_canonical_name("unknown_tool") == "unknown_tool"
        assert get_canonical_name("custom_tool") == "custom_tool"
        assert get_canonical_name("my_special_tool") == "my_special_tool"

    def test_empty_string_returns_empty(self):
        """Empty string returns empty string."""
        assert get_canonical_name("") == ""

    def test_case_sensitive(self):
        """Name matching is case-sensitive."""
        # Uppercase alias should not match
        assert get_canonical_name("EXECUTE_BASH") == "EXECUTE_BASH"
        assert get_canonical_name("Read_File") == "Read_File"


class TestGetAliases:
    """Tests for get_aliases() function."""

    def test_returns_aliases_for_canonical(self):
        """Returns aliases for canonical names with aliases."""
        aliases = get_aliases("shell")
        assert "execute_bash" in aliases
        assert "shell_readonly" not in aliases
        assert "bash" in aliases
        assert "run" in aliases

    def test_returns_empty_for_no_aliases(self):
        """Returns empty set for canonical names without aliases."""
        assert get_aliases("docker") == set()
        assert get_aliases("cicd") == set()

    def test_returns_empty_for_unknown(self):
        """Returns empty set for unknown names."""
        assert get_aliases("unknown_tool") == set()
        assert get_aliases("") == set()


class TestIsValidToolName:
    """Tests for is_valid_tool_name() function."""

    def test_canonical_names_valid(self):
        """Canonical names are valid."""
        assert is_valid_tool_name("shell") is True
        assert is_valid_tool_name("read") is True
        assert is_valid_tool_name("edit") is True
        assert is_valid_tool_name("git") is True

    def test_aliases_valid(self):
        """Aliases are valid."""
        assert is_valid_tool_name("execute_bash") is True
        assert is_valid_tool_name("read_file") is True
        assert is_valid_tool_name("edit_files") is True
        assert is_valid_tool_name("git_status") is True
        assert is_valid_tool_name("shell_readonly") is False

    def test_unknown_names_invalid(self):
        """Unknown names are invalid."""
        assert is_valid_tool_name("unknown_tool") is False
        assert is_valid_tool_name("not_a_tool") is False

    def test_empty_string_invalid(self):
        """Empty string is invalid."""
        assert is_valid_tool_name("") is False


class TestGetAllCanonicalNames:
    """Tests for get_all_canonical_names() function."""

    def test_returns_set(self):
        """Returns a set of names."""
        names = get_all_canonical_names()
        assert isinstance(names, set)

    def test_contains_core_tools(self):
        """Contains core filesystem tools."""
        names = get_all_canonical_names()
        assert "read" in names
        assert "write" in names
        assert "edit" in names
        assert "shell" in names

    def test_contains_all_toolnames_attributes(self):
        """Contains all ToolNames class attributes."""
        names = get_all_canonical_names()
        assert "git" in names
        assert "grep" in names
        assert "test" in names
        assert "docker" in names

    def test_does_not_contain_aliases(self):
        """Does not contain alias names."""
        names = get_all_canonical_names()
        assert "execute_bash" not in names
        assert "shell_readonly" not in names
        assert "read_file" not in names
        assert "edit_files" not in names


class TestGetNameMapping:
    """Tests for get_name_mapping() function."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        mapping = get_name_mapping()
        assert isinstance(mapping, dict)

    def test_canonical_maps_to_itself(self):
        """Canonical names map to themselves."""
        mapping = get_name_mapping()
        assert mapping["shell"] == "shell"
        assert mapping["read"] == "read"
        assert mapping["edit"] == "edit"

    def test_aliases_map_to_canonical(self):
        """Aliases map to their canonical names."""
        mapping = get_name_mapping()
        assert mapping["execute_bash"] == "shell"
        assert "shell_readonly" not in mapping
        assert mapping["read_file"] == "read"
        assert mapping["edit_files"] == "edit"

    def test_contains_all_names(self):
        """Contains both canonical names and aliases."""
        mapping = get_name_mapping()
        # Canonical
        assert "git" in mapping
        assert "shell" in mapping
        # Aliases
        assert "execute_bash" in mapping
        assert "git_status" in mapping


class TestCanonicalizeToolSet:
    """Tests for canonicalize_tool_set() function."""

    def test_converts_aliases_to_canonical(self):
        """Converts alias names to canonical form."""
        tools = {"read_file", "execute_bash", "edit_files"}
        result = canonicalize_tool_set(tools)
        assert result == {"read", "shell", "edit"}

    def test_preserves_canonical_names(self):
        """Preserves already canonical names."""
        tools = {"read", "shell", "edit"}
        result = canonicalize_tool_set(tools)
        assert result == {"read", "shell", "edit"}

    def test_mixed_names(self):
        """Handles mix of canonical and alias names."""
        tools = {"read", "execute_bash", "edit_files"}
        result = canonicalize_tool_set(tools)
        assert result == {"read", "shell", "edit"}

    def test_duplicates_collapse(self):
        """Duplicate canonical names collapse to one."""
        tools = {"execute_bash", "bash", "run"}  # All map to "shell"
        result = canonicalize_tool_set(tools)
        assert result == {"shell"}

    def test_empty_set(self):
        """Handles empty set."""
        result = canonicalize_tool_set(set())
        assert result == set()

    def test_unknown_names_preserved(self):
        """Unknown names are preserved as-is."""
        tools = {"read", "custom_tool", "my_special_tool"}
        result = canonicalize_tool_set(tools)
        assert result == {"read", "custom_tool", "my_special_tool"}


class TestCanonicalizeToolDict:
    """Tests for canonicalize_tool_dict() function."""

    def test_converts_keys_to_canonical(self):
        """Converts dictionary keys to canonical form."""
        mapping = {"read_file": 0.8, "execute_bash": 0.5}
        result = canonicalize_tool_dict(mapping)
        assert result == {"read": 0.8, "shell": 0.5}

    def test_preserves_values(self):
        """Values are preserved unchanged."""
        mapping = {"read_file": {"score": 0.9, "count": 5}}
        result = canonicalize_tool_dict(mapping)
        assert result["read"] == {"score": 0.9, "count": 5}

    def test_preserves_canonical_keys(self):
        """Already canonical keys are preserved."""
        mapping = {"read": 1.0, "shell": 0.5}
        result = canonicalize_tool_dict(mapping)
        assert result == {"read": 1.0, "shell": 0.5}

    def test_empty_dict(self):
        """Handles empty dictionary."""
        result = canonicalize_tool_dict({})
        assert result == {}

    def test_duplicate_keys_last_wins(self):
        """When aliases map to same canonical, last value wins."""
        # Note: dict preserves insertion order, so "cmd" comes last
        mapping = {"execute_bash": 0.5, "bash": 0.6, "cmd": 0.7}
        result = canonicalize_tool_dict(mapping)
        # All map to "shell", last one (cmd: 0.7) wins
        assert result == {"shell": 0.7}


class TestCanonicalizeToolList:
    """Tests for canonicalize_tool_list() function."""

    def test_converts_to_canonical(self):
        """Converts alias names to canonical form."""
        tools = ["read_file", "edit_files", "run_tests"]
        result = canonicalize_tool_list(tools)
        assert result == ["read", "edit", "test"]

    def test_preserves_order(self):
        """Preserves list order."""
        tools = ["execute_bash", "read_file", "edit_files"]
        result = canonicalize_tool_list(tools)
        assert result == ["shell", "read", "edit"]

    def test_preserves_duplicates(self):
        """Preserves duplicate entries unlike set."""
        tools = ["read_file", "read", "read_file"]
        result = canonicalize_tool_list(tools)
        assert result == ["read", "read", "read"]

    def test_empty_list(self):
        """Handles empty list."""
        result = canonicalize_tool_list([])
        assert result == []

    def test_unknown_names_preserved(self):
        """Unknown names are preserved."""
        tools = ["read", "custom_tool", "edit"]
        result = canonicalize_tool_list(tools)
        assert result == ["read", "custom_tool", "edit"]


class TestCanonicalizeTransitions:
    """Tests for canonicalize_transitions() function."""

    def test_canonicalizes_outer_keys(self):
        """Canonicalizes outer dictionary keys."""
        transitions = {
            "read_file": [("edit", 0.4)],
        }
        result = canonicalize_transitions(transitions)
        assert "read" in result
        assert "read_file" not in result

    def test_canonicalizes_inner_tools(self):
        """Canonicalizes tool names in tuples."""
        transitions = {
            "read": [("edit_files", 0.4), ("execute_bash", 0.3)],
        }
        result = canonicalize_transitions(transitions)
        assert result["read"] == [("edit", 0.4), ("shell", 0.3)]

    def test_preserves_probabilities(self):
        """Preserves probability values."""
        transitions = {
            "read_file": [("edit_files", 0.4), ("run_tests", 0.6)],
        }
        result = canonicalize_transitions(transitions)
        assert result["read"] == [("edit", 0.4), ("test", 0.6)]

    def test_full_canonicalization(self):
        """Full canonicalization of complex transitions."""
        transitions = {
            "read_file": [("edit_files", 0.4), ("execute_bash", 0.3)],
            "edit_files": [("run_tests", 0.5), ("git_commit", 0.2)],
        }
        result = canonicalize_transitions(transitions)
        expected = {
            "read": [("edit", 0.4), ("shell", 0.3)],
            "edit": [("test", 0.5), ("git", 0.2)],
        }
        assert result == expected

    def test_empty_transitions(self):
        """Handles empty transitions dict."""
        result = canonicalize_transitions({})
        assert result == {}

    def test_empty_next_tools(self):
        """Handles empty next_tools list."""
        transitions = {"read_file": []}
        result = canonicalize_transitions(transitions)
        assert result == {"read": []}


class TestCanonicalizeDependencies:
    """Tests for canonicalize_dependencies() function."""

    def test_canonicalizes_tool_name(self):
        """Canonicalizes main tool_name."""
        deps = [
            ToolDependency(
                tool_name="edit_files",
                depends_on={"read"},
                enables={"test"},
                weight=0.9,
            )
        ]
        result = canonicalize_dependencies(deps)
        assert result[0].tool_name == "edit"

    def test_canonicalizes_depends_on(self):
        """Canonicalizes depends_on set."""
        deps = [
            ToolDependency(
                tool_name="edit",
                depends_on={"read_file", "execute_bash"},
                enables=set(),
                weight=1.0,
            )
        ]
        result = canonicalize_dependencies(deps)
        assert result[0].depends_on == {"read", "shell"}

    def test_canonicalizes_enables(self):
        """Canonicalizes enables set."""
        deps = [
            ToolDependency(
                tool_name="edit",
                depends_on=set(),
                enables={"run_tests", "git_commit"},
                weight=1.0,
            )
        ]
        result = canonicalize_dependencies(deps)
        assert result[0].enables == {"test", "git"}

    def test_preserves_weight(self):
        """Preserves weight value."""
        deps = [
            ToolDependency(
                tool_name="edit_files",
                depends_on=set(),
                enables=set(),
                weight=0.75,
            )
        ]
        result = canonicalize_dependencies(deps)
        assert result[0].weight == 0.75

    def test_full_canonicalization(self):
        """Full canonicalization of dependency."""
        deps = [
            ToolDependency(
                tool_name="edit_files",
                depends_on={"read_file"},
                enables={"run_tests"},
                weight=0.9,
            )
        ]
        result = canonicalize_dependencies(deps)
        assert result[0].tool_name == "edit"
        assert result[0].depends_on == {"read"}
        assert result[0].enables == {"test"}
        assert result[0].weight == 0.9

    def test_multiple_dependencies(self):
        """Handles multiple dependencies."""
        deps = [
            ToolDependency(tool_name="edit_files", depends_on={"read_file"}, enables=set()),
            ToolDependency(tool_name="run_tests", depends_on={"edit_files"}, enables=set()),
        ]
        result = canonicalize_dependencies(deps)
        assert len(result) == 2
        assert result[0].tool_name == "edit"
        assert result[1].tool_name == "test"

    def test_empty_dependencies(self):
        """Handles empty dependencies list."""
        result = canonicalize_dependencies([])
        assert result == []


class TestValidateToolNames:
    """Tests for validate_tool_names() function."""

    def test_returns_empty_for_canonical_only(self):
        """Returns empty list when all names are canonical."""
        tools = {"read", "shell", "edit"}
        result = validate_tool_names(tools, warn=False)
        assert result == []

    def test_returns_legacy_names(self):
        """Returns list of legacy (alias) names found."""
        tools = {"read", "execute_bash", "edit_files"}
        result = validate_tool_names(tools, warn=False)
        assert "execute_bash" in result
        assert "edit_files" in result
        assert "read" not in result

    def test_accepts_set_input(self):
        """Accepts set input."""
        tools = {"read_file", "edit"}
        result = validate_tool_names(tools, warn=False)
        assert result == ["read_file"]

    def test_accepts_list_input(self):
        """Accepts list input."""
        tools = ["read_file", "edit", "execute_bash"]
        result = validate_tool_names(tools, warn=False)
        assert set(result) == {"read_file", "execute_bash"}

    def test_accepts_dict_input(self):
        """Accepts dict input (uses keys)."""
        tools = {"read_file": 0.5, "edit": 0.3, "execute_bash": 0.2}
        result = validate_tool_names(tools, warn=False)
        assert set(result) == {"read_file", "execute_bash"}

    def test_logs_warning_when_enabled(self, caplog):
        """Logs warnings when warn=True."""
        tools = {"execute_bash"}
        with caplog.at_level(logging.WARNING):
            validate_tool_names(tools, context="test context", warn=True)
        assert "Legacy tool name 'execute_bash' in test context" in caplog.text
        assert "use 'shell' instead" in caplog.text

    def test_no_warning_when_disabled(self, caplog):
        """No warnings when warn=False."""
        tools = {"execute_bash"}
        with caplog.at_level(logging.WARNING):
            validate_tool_names(tools, warn=False)
        assert "execute_bash" not in caplog.text

    def test_context_in_warning(self, caplog):
        """Context appears in warning message."""
        tools = {"read_file"}
        with caplog.at_level(logging.WARNING):
            validate_tool_names(tools, context="coding config", warn=True)
        assert "in coding config" in caplog.text

    def test_no_context_in_warning(self, caplog):
        """Warning works without context."""
        tools = {"read_file"}
        with caplog.at_level(logging.WARNING):
            validate_tool_names(tools, warn=True)
        assert "Legacy tool name 'read_file'" in caplog.text

    def test_empty_input(self):
        """Handles empty input."""
        assert validate_tool_names(set(), warn=False) == []
        assert validate_tool_names([], warn=False) == []
        assert validate_tool_names({}, warn=False) == []


class TestGetLegacyNamesReport:
    """Tests for get_legacy_names_report() function."""

    def test_returns_mapping_of_legacy_to_canonical(self):
        """Returns dict mapping legacy names to canonical."""
        tools = {"read", "execute_bash", "edit_files"}
        result = get_legacy_names_report(tools)
        assert result == {"execute_bash": "shell", "edit_files": "edit"}

    def test_empty_for_canonical_only(self):
        """Returns empty dict when all canonical."""
        tools = {"read", "shell", "edit"}
        result = get_legacy_names_report(tools)
        assert result == {}

    def test_accepts_set_input(self):
        """Accepts set input."""
        tools = {"read_file"}
        result = get_legacy_names_report(tools)
        assert result == {"read_file": "read"}

    def test_accepts_list_input(self):
        """Accepts list input."""
        tools = ["read_file", "run_tests"]
        result = get_legacy_names_report(tools)
        assert result == {"read_file": "read", "run_tests": "test"}

    def test_accepts_dict_input(self):
        """Accepts dict input (uses keys)."""
        tools = {"read_file": 1, "shell": 2}
        result = get_legacy_names_report(tools)
        assert result == {"read_file": "read"}

    def test_empty_input(self):
        """Handles empty input."""
        assert get_legacy_names_report(set()) == {}
        assert get_legacy_names_report([]) == {}
        assert get_legacy_names_report({}) == {}

    def test_no_warnings_logged(self, caplog):
        """Does not log any warnings."""
        tools = {"execute_bash", "read_file"}
        with caplog.at_level(logging.WARNING):
            get_legacy_names_report(tools)
        assert caplog.text == ""


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_special_characters_in_unknown_names(self):
        """Unknown names with special characters preserved."""
        tools = {"my-tool", "tool_v2", "tool.name"}
        result = canonicalize_tool_set(tools)
        assert result == {"my-tool", "tool_v2", "tool.name"}

    def test_numeric_suffix_names(self):
        """Names with numeric suffixes preserved."""
        tools = {"tool1", "tool2", "read"}
        result = canonicalize_tool_set(tools)
        assert "tool1" in result
        assert "tool2" in result

    def test_whitespace_names_preserved(self):
        """Whitespace in names preserved (not recommended)."""
        # This tests that we don't strip or modify names
        tools = {" read ", "  shell"}
        result = canonicalize_tool_set(tools)
        assert " read " in result
        assert "  shell" in result

    def test_unicode_names_preserved(self):
        """Unicode names are preserved."""
        tools = {"read", "tool_unicode_test"}
        result = canonicalize_tool_set(tools)
        assert "tool_unicode_test" in result

    def test_very_long_names_preserved(self):
        """Very long names are preserved."""
        long_name = "a" * 1000
        tools = {long_name, "read"}
        result = canonicalize_tool_set(tools)
        assert long_name in result


class TestReExports:
    """Tests verifying re-exports from victor.tools.tool_names."""

    def test_toolnames_exported(self):
        """ToolNames class is exported."""
        from victor.framework.tool_naming import ToolNames

        assert ToolNames.READ == "read"

    def test_tool_aliases_exported(self):
        """TOOL_ALIASES dict is exported."""
        from victor.framework.tool_naming import TOOL_ALIASES

        assert "execute_bash" in TOOL_ALIASES

    def test_canonical_to_aliases_exported(self):
        """CANONICAL_TO_ALIASES dict is exported."""
        from victor.framework.tool_naming import CANONICAL_TO_ALIASES

        assert "shell" in CANONICAL_TO_ALIASES

    def test_get_canonical_name_exported(self):
        """get_canonical_name function is exported."""
        from victor.framework.tool_naming import get_canonical_name

        assert get_canonical_name("execute_bash") == "shell"

    def test_get_aliases_exported(self):
        """get_aliases function is exported."""
        from victor.framework.tool_naming import get_aliases

        assert "execute_bash" in get_aliases("shell")

    def test_is_valid_tool_name_exported(self):
        """is_valid_tool_name function is exported."""
        from victor.framework.tool_naming import is_valid_tool_name

        assert is_valid_tool_name("shell") is True

    def test_get_all_canonical_names_exported(self):
        """get_all_canonical_names function is exported."""
        from victor.framework.tool_naming import get_all_canonical_names

        names = get_all_canonical_names()
        assert "read" in names

    def test_get_name_mapping_exported(self):
        """get_name_mapping function is exported."""
        from victor.framework.tool_naming import get_name_mapping

        mapping = get_name_mapping()
        assert mapping["execute_bash"] == "shell"

    def test_toolnameentry_exported(self):
        """ToolNameEntry class is exported."""
        from victor.framework.tool_naming import ToolNameEntry

        entry = ToolNameEntry(canonical="test")
        assert entry.canonical == "test"
