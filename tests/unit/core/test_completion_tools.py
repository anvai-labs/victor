"""Unit tests for the caller-declared completion-tool policy.

The declaration is generic: the framework only learns that a named tool's
batch must be allowed to execute, never what the tool does. Empty
declaration = every query False (default-off parity).
"""

from types import SimpleNamespace

from victor.core.completion_tools import (
    CompletionToolPolicy,
    normalize_completion_tool_names,
    resolve_completion_tool_names,
)


class TestNormalizeCompletionToolNames:
    def test_empty_declaration_normalizes_to_empty(self):
        assert normalize_completion_tool_names(()) == ()
        assert normalize_completion_tool_names([]) == ()
        assert normalize_completion_tool_names(None) == ()

    def test_non_string_and_blank_entries_dropped(self):
        assert normalize_completion_tool_names(["submit", 7, None, "  "]) == ("submit",)

    def test_duplicates_collapse(self):
        assert normalize_completion_tool_names(["submit", "submit"]) == ("submit",)

    def test_alias_spelling_canonicalizes(self):
        assert normalize_completion_tool_names(["ReadFile"]) == ("read_file",)


class TestCompletionToolPolicy:
    def test_empty_policy_disabled_and_never_critical(self):
        policy = CompletionToolPolicy()
        assert policy.enabled is False
        assert policy.is_completion_critical("submit") is False
        assert policy.exempt_block_reason("submit", "blocked: dup") == "blocked: dup"

    def test_declared_tool_is_completion_critical(self):
        policy = CompletionToolPolicy(["submit_proposal"])
        assert policy.enabled is True
        assert policy.is_completion_critical("submit_proposal") is True
        assert policy.is_completion_critical("submit_proposal ") is True
        assert policy.is_completion_critical("other_tool") is False

    def test_case_insensitive_match(self):
        policy = CompletionToolPolicy(["Submit"])
        assert policy.is_completion_critical("submit") is True
        assert policy.is_completion_critical("SUBMIT") is True

    def test_non_string_name_never_critical(self):
        policy = CompletionToolPolicy(["submit"])
        assert policy.is_completion_critical(None) is False
        assert policy.is_completion_critical(7) is False

    def test_exempt_block_reason_clears_only_declared_tools(self):
        policy = CompletionToolPolicy(["submit"])
        assert policy.exempt_block_reason("submit", "blocked after warning") is None
        assert policy.exempt_block_reason("read", "blocked after warning") == (
            "blocked after warning"
        )
        assert policy.exempt_block_reason("submit", None) is None


class TestResolveCompletionToolNames:
    def test_none_settings_resolve_empty(self):
        assert resolve_completion_tool_names(None) == ()

    def test_nested_tool_settings_resolution(self):
        settings = SimpleNamespace(tools=SimpleNamespace(completion_tools=["submit_proposal"]))
        assert "submit_proposal" in resolve_completion_tool_names(settings)

    def test_legacy_flat_resolution(self):
        settings = SimpleNamespace(completion_tools=["submit_proposal"])
        assert "submit_proposal" in resolve_completion_tool_names(settings)

    def test_missing_declaration_resolves_empty(self):
        assert resolve_completion_tool_names(SimpleNamespace()) == ()
        assert resolve_completion_tool_names(SimpleNamespace(tools=None)) == ()
