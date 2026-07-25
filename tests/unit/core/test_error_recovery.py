"""
Tests for the Error Recovery Chain of Responsibility.

These tests verify the SOLID-compliant error recovery system that handles
tool execution failures gracefully.
"""

import pytest
from victor.agent.error_recovery import (
    RecoveryAction,
    RecoveryResult,
    ErrorRecoveryHandler,
    MissingParameterHandler,
    ToolNotFoundHandler,
    NetworkErrorHandler,
    FileNotFoundHandler,
    RateLimitHandler,
    PermissionErrorHandler,
    TypeErrorHandler,
    build_recovery_chain,
    get_recovery_chain,
    recover_from_error,
)


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_should_retry_true_for_retry_actions(self):
        """Test that retry variants return True for should_retry."""
        result = RecoveryResult(action=RecoveryAction.RETRY)
        assert result.should_retry is True

        result = RecoveryResult(action=RecoveryAction.RETRY_WITH_DEFAULTS)
        assert result.should_retry is True

        result = RecoveryResult(action=RecoveryAction.RETRY_WITH_INFERRED)
        assert result.should_retry is True

    def test_should_retry_false_for_non_retry_actions(self):
        """Test that non-retry actions return False for should_retry."""
        assert RecoveryResult(action=RecoveryAction.SKIP).should_retry is False
        assert RecoveryResult(action=RecoveryAction.ABORT).should_retry is False
        assert RecoveryResult(action=RecoveryAction.FALLBACK_TOOL).should_retry is False
        assert RecoveryResult(action=RecoveryAction.ASK_USER).should_retry is False

    def test_can_retry_with_retries_remaining(self):
        """Test can_retry when retries are remaining."""
        result = RecoveryResult(action=RecoveryAction.RETRY, retry_count=0, max_retries=3)
        assert result.can_retry is True

        result = RecoveryResult(action=RecoveryAction.RETRY, retry_count=2, max_retries=3)
        assert result.can_retry is True

    def test_can_retry_false_when_exhausted(self):
        """Test can_retry when retries are exhausted."""
        result = RecoveryResult(action=RecoveryAction.RETRY, retry_count=3, max_retries=3)
        assert result.can_retry is False

        result = RecoveryResult(action=RecoveryAction.RETRY, retry_count=5, max_retries=3)
        assert result.can_retry is False


class TestMissingParameterHandler:
    """Tests for MissingParameterHandler."""

    @pytest.fixture
    def handler(self):
        return MissingParameterHandler()

    def test_can_handle_missing_argument_error(self, handler):
        """Test detection of missing argument errors."""
        error = Exception("missing 1 required positional argument: 'file_path'")
        assert handler.can_handle(error, "read", {}) is True

    def test_can_handle_missing_parameter_error(self, handler):
        """Test detection of missing parameter errors."""
        error = Exception("required parameter 'path' was not provided")
        assert handler.can_handle(error, "ls", {}) is True

    def test_can_handle_required_property_error(self, handler):
        """Test detection of required property errors."""
        error = Exception("'limit' is a required property")
        assert handler.can_handle(error, "search", {}) is True

    def test_cannot_handle_unrelated_error(self, handler):
        """Test that unrelated errors are not handled."""
        error = Exception("File not found")
        assert handler.can_handle(error, "read", {}) is False

    def test_handle_provides_default_for_known_param(self, handler):
        """Test that known parameters get default values."""
        error = Exception("missing 1 required positional argument: 'file_path'")
        result = handler.handle(error, "read", {"content": "test"})

        assert result.action == RecoveryAction.RETRY_WITH_DEFAULTS
        assert result.modified_args["file_path"] == "."
        assert result.modified_args["content"] == "test"

    def test_handle_provides_default_for_path(self, handler):
        """Test default value for 'path' parameter."""
        error = Exception("missing required argument: path")
        result = handler.handle(error, "ls", {})

        assert result.action == RecoveryAction.RETRY_WITH_DEFAULTS
        assert result.modified_args["path"] == "."

    def test_handle_recovers_wrapped_value_payload_for_write(self, handler):
        """Wrapped write payloads should be recovered before default retries."""
        error = Exception("Error: Missing required arguments: path, content")
        result = handler.handle(
            error,
            "write",
            {
                "value": (
                    '{"file_path":"victor/framework/graph_protocols.py",'
                    '"text":"print(\\"hi\\")\\n"}'
                )
            },
        )

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args == {
            "path": "victor/framework/graph_protocols.py",
            "content": 'print("hi")\n',
        }

    def test_handle_skips_when_multi_param_error_has_unknown_values(self, handler):
        """Multi-parameter missing errors should not retry with partial defaults."""
        error = Exception("Error: Missing required arguments: path, content")
        result = handler.handle(error, "write", {})

        assert result.action == RecoveryAction.SKIP
        assert "path, content" in result.user_message

    def test_handle_provides_default_for_limit(self, handler):
        """Test default value for 'limit' parameter."""
        error = Exception("'limit' is a required property")
        result = handler.handle(error, "search", {"query": "test"})

        assert result.action == RecoveryAction.RETRY_WITH_DEFAULTS
        assert result.modified_args["limit"] == 100

    def test_handle_skips_unknown_param(self, handler):
        """Test that unknown parameters result in skip."""
        error = Exception("missing 1 required positional argument: 'unknown_param'")
        result = handler.handle(error, "custom_tool", {})

        assert result.action == RecoveryAction.SKIP
        assert "unknown_param" in result.user_message


class TestToolNotFoundHandler:
    """Tests for ToolNotFoundHandler."""

    @pytest.fixture
    def handler(self):
        return ToolNotFoundHandler()

    def test_can_handle_not_found_error(self, handler):
        """Test detection of tool not found errors."""
        error = Exception("Tool 'symbol' not found")
        assert handler.can_handle(error, "symbol", {}) is True

    def test_can_handle_unknown_tool_error(self, handler):
        """Test detection of unknown tool errors."""
        error = Exception("Unknown tool: semantic_search")
        assert handler.can_handle(error, "semantic_search", {}) is True

    def test_can_handle_unregistered_error(self, handler):
        """Test detection of unregistered tool errors."""
        error = Exception("Tool 'tree' is unregistered")
        assert handler.can_handle(error, "tree", {}) is True

    def test_cannot_handle_unrelated_error(self, handler):
        """Test that unrelated errors are not handled."""
        error = Exception("Network timeout")
        assert handler.can_handle(error, "web_search", {}) is False

    def test_handle_provides_fallback_for_symbol(self, handler):
        """Test fallback from symbol to grep."""
        error = Exception("Tool 'symbol' not found")
        result = handler.handle(error, "symbol", {"symbol_name": "MyClass"})

        assert result.action == RecoveryAction.FALLBACK_TOOL
        assert result.fallback_tool == "grep"

    def test_handle_provides_fallback_for_tree(self, handler):
        """Test fallback from tree to ls."""
        error = Exception("Tool not found")
        result = handler.handle(error, "tree", {"path": "."})

        assert result.action == RecoveryAction.FALLBACK_TOOL
        assert result.fallback_tool == "ls"

    def test_handle_skips_unknown_tool(self, handler):
        """Test that unknown tools result in skip."""
        error = Exception("Tool not found")
        result = handler.handle(error, "custom_tool", {})

        assert result.action == RecoveryAction.SKIP
        assert "no fallback" in result.user_message.lower()


class TestNetworkErrorHandler:
    """Tests for NetworkErrorHandler."""

    @pytest.fixture
    def handler(self):
        return NetworkErrorHandler()

    def test_can_handle_timeout_error(self, handler):
        """Test detection of timeout errors."""
        error = Exception("Connection timeout after 30s")
        assert handler.can_handle(error, "web_fetch", {}) is True

    def test_can_handle_connection_error(self, handler):
        """Test detection of connection errors."""
        error = Exception("Connection refused")
        assert handler.can_handle(error, "web_fetch", {}) is True

    def test_can_handle_network_error(self, handler):
        """Test detection of network errors."""
        error = Exception("Network unreachable")
        assert handler.can_handle(error, "web_search", {}) is True

    def test_can_handle_dns_error(self, handler):
        """Test detection of DNS errors."""
        error = Exception("DNS resolution failed")
        assert handler.can_handle(error, "web_fetch", {}) is True

    def test_can_handle_ssl_error(self, handler):
        """Test detection of SSL errors."""
        error = Exception("SSL certificate verification failed")
        assert handler.can_handle(error, "web_fetch", {}) is True

    def test_cannot_handle_unrelated_error(self, handler):
        """Test that unrelated errors are not handled."""
        error = Exception("File not found")
        assert handler.can_handle(error, "read", {}) is False

    def test_handle_returns_retry(self, handler):
        """Test that network errors result in retry."""
        error = Exception("Connection timeout")
        result = handler.handle(error, "web_fetch", {"url": "http://example.com"})

        assert result.action == RecoveryAction.RETRY
        assert result.max_retries == 3
        assert "retry" in result.user_message.lower()


class TestFileNotFoundHandler:
    """Tests for FileNotFoundHandler."""

    @pytest.fixture
    def handler(self):
        return FileNotFoundHandler()

    def test_can_handle_file_not_found_error(self, handler):
        """Test detection of file not found errors."""
        error = Exception("File not found: test.py")
        assert handler.can_handle(error, "read", {}) is True

    def test_can_handle_no_such_file_error(self, handler):
        """Test detection of no such file errors."""
        error = Exception("No such file or directory")
        assert handler.can_handle(error, "read", {}) is True

    def test_can_handle_does_not_exist_error(self, handler):
        """Test detection of does not exist errors."""
        error = Exception("Path does not exist")
        assert handler.can_handle(error, "read", {}) is True

    def test_can_handle_filenotfounderror(self, handler):
        """Test detection of FileNotFoundError."""
        error = FileNotFoundError("test.py")
        assert handler.can_handle(error, "read", {}) is True

    def test_cannot_handle_unrelated_error(self, handler):
        """Test that unrelated errors are not handled."""
        error = Exception("Permission denied")
        assert handler.can_handle(error, "read", {}) is False

    def test_handle_tries_path_variation(self, handler):
        """Test that file not found tries path variations."""
        error = Exception("File not found")
        result = handler.handle(error, "read", {"path": "./test.py"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["path"] == "test.py"

    def test_handle_tries_init_file(self, handler):
        """Test that module path tries __init__.py."""
        error = Exception("File not found")
        result = handler.handle(error, "read", {"path": "mymodule.py"})

        # Should try variations
        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert "tried_variations" in result.metadata

    def test_handle_prefers_package_file_suggestion_for_reads(self, handler):
        """Read retries should prefer concrete package files over directory suggestions."""
        error = Exception(
            "File not found: victor/core/registry.py\n"
            "Did you mean one of these?\n"
            "  - victor/core/registry/\n"
            "  - victor/core/registry/base.py\n"
            "  - victor/core/registry_base.py"
        )

        result = handler.handle(error, "read", {"path": "victor/core/registry.py"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["path"] == "victor/core/registry/base.py"
        assert result.metadata["suggested_paths"][0] == "victor/core/registry/"

    def test_handle_uses_file_arg_key_when_retrying_suggestion(self, handler):
        """Suggestion retries should preserve the original path argument key."""
        error = Exception(
            "File not found: setings.yaml\nDid you mean one of these?\n  - settings.yaml"
        )

        result = handler.handle(error, "read", {"file": "setings.yaml"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["file"] == "settings.yaml"

    def test_handle_skips_weak_suggestion_for_different_document(self, handler):
        """A dissimilar 'Did you mean' match must NOT silently substitute the file.

        Regression: session proximaDB-5b2726a3 — read of a file absent from the
        checked-out branch was silently answered with a different document, then
        a sticky redirect made the not-found state unobservable to the model.
        """
        error = Exception(
            "File not found: docs/12-design/MODALITY_COMPILE_TIME_COMPOSITION_2026_07_24.adoc\n"
            "Did you mean one of these?\n"
            "  - docs/12-design/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc"
        )

        result = handler.handle(
            error,
            "read",
            {"path": "docs/12-design/MODALITY_COMPILE_TIME_COMPOSITION_2026_07_24.adoc"},
        )

        assert result.action == RecoveryAction.SKIP
        assert result.metadata["suggested_paths"] == [
            "docs/12-design/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc"
        ]

    def test_handle_retries_typo_class_suggestion(self, handler):
        """A near-identical basename (typo/date slip) is safe to auto-retry."""
        error = Exception(
            "File not found: docs/TRACKER_2026_07_22.adoc\n"
            "Did you mean one of these?\n"
            "  - docs/TRACKER_2026_07_21.adoc"
        )

        result = handler.handle(error, "read", {"path": "docs/TRACKER_2026_07_22.adoc"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["path"] == "docs/TRACKER_2026_07_21.adoc"

    def test_handle_rejects_self_suggestion(self, handler):
        """Recovery should not retry the same unresolved path."""
        error = Exception("File not found: Cargo.toml\nDid you mean one of these?\n  - Cargo.toml")

        result = handler.handle(error, "read", {"path": "Cargo.toml"})

        assert result.action == RecoveryAction.SKIP
        assert result.metadata["suggested_paths"] == ["Cargo.toml"]

    def test_get_path_variations_removes_leading_dot(self, handler):
        """Test path variation removes leading ./"""
        variations = handler._get_path_variations("./test.py")
        assert "test.py" in variations

    def test_get_path_variations_adds_leading_dot(self, handler):
        """Test path variation adds leading ./"""
        variations = handler._get_path_variations("test.py")
        assert "./test.py" in variations


class TestRateLimitHandler:
    """Tests for RateLimitHandler."""

    @pytest.fixture
    def handler(self):
        return RateLimitHandler()

    def test_can_handle_rate_limit_error(self, handler):
        """Test detection of rate limit errors."""
        error = Exception("Rate limit exceeded")
        assert handler.can_handle(error, "web_search", {}) is True

    def test_can_handle_429_error(self, handler):
        """Test detection of 429 status code errors."""
        error = Exception("HTTP 429 Too Many Requests")
        assert handler.can_handle(error, "web_search", {}) is True

    def test_can_handle_throttle_error(self, handler):
        """Test detection of throttle errors."""
        error = Exception("Request throttled")
        assert handler.can_handle(error, "api_call", {}) is True

    def test_can_handle_quota_exceeded_error(self, handler):
        """Test detection of quota exceeded errors."""
        error = Exception("API quota exceeded")
        assert handler.can_handle(error, "web_search", {}) is True

    def test_cannot_handle_unrelated_error(self, handler):
        """Test that unrelated errors are not handled."""
        error = Exception("File not found")
        assert handler.can_handle(error, "read", {}) is False

    def test_handle_returns_retry_with_delay(self, handler):
        """Test that rate limit errors result in retry with delay."""
        error = Exception("Rate limit exceeded")
        result = handler.handle(error, "web_search", {})

        assert result.action == RecoveryAction.RETRY
        assert result.max_retries == 3
        assert "retry_delay_seconds" in result.metadata

    def test_handle_extracts_retry_after(self, handler):
        """Test extraction of retry-after header."""
        error = Exception("Rate limited. Retry-After: 30")
        result = handler.handle(error, "web_search", {})

        assert result.metadata["retry_delay_seconds"] == 30.0


class TestPermissionErrorHandler:
    """Tests for PermissionErrorHandler."""

    @pytest.fixture
    def handler(self):
        return PermissionErrorHandler()

    def test_can_handle_permission_denied(self, handler):
        """Test detection of permission denied errors."""
        error = Exception("Permission denied")
        assert handler.can_handle(error, "write", {}) is True

    def test_can_handle_access_denied(self, handler):
        """Test detection of access denied errors."""
        error = Exception("Access denied")
        assert handler.can_handle(error, "write", {}) is True

    def test_can_handle_forbidden(self, handler):
        """Test detection of forbidden errors."""
        error = Exception("403 Forbidden")
        assert handler.can_handle(error, "write", {}) is True

    def test_can_handle_permissionerror(self, handler):
        """Test detection of PermissionError."""
        error = PermissionError("Cannot write to file")
        assert handler.can_handle(error, "write", {}) is True

    def test_cannot_handle_unrelated_error(self, handler):
        """Test that unrelated errors are not handled."""
        error = Exception("File not found")
        assert handler.can_handle(error, "read", {}) is False

    def test_handle_asks_user(self, handler):
        """Test that permission errors ask user."""
        error = Exception("Permission denied")
        result = handler.handle(error, "write", {"path": "/etc/passwd"})

        assert result.action == RecoveryAction.ASK_USER
        assert "permission" in result.user_message.lower()


class TestTypeErrorHandler:
    """Tests for TypeErrorHandler."""

    @pytest.fixture
    def handler(self):
        return TypeErrorHandler()

    def test_can_handle_typeerror(self, handler):
        """Test detection of TypeError."""
        error = TypeError("expected str, got int")
        assert handler.can_handle(error, "search", {}) is True

    def test_can_handle_type_in_message(self, handler):
        """Test detection of type errors in message."""
        error = Exception("Invalid type for parameter")
        assert handler.can_handle(error, "search", {}) is True

    def test_cannot_handle_unrelated_error(self, handler):
        """Test that unrelated errors are not handled."""
        error = Exception("File not found")
        assert handler.can_handle(error, "read", {}) is False

    def test_handle_converts_string_bool_true(self, handler):
        """Test conversion of string 'true' to bool."""
        error = TypeError("expected bool, got str")
        result = handler.handle(error, "search", {"recursive": "true"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["recursive"] is True

    def test_handle_converts_string_bool_false(self, handler):
        """Test conversion of string 'false' to bool."""
        error = TypeError("expected bool")
        result = handler.handle(error, "search", {"recursive": "false"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["recursive"] is False

    def test_handle_converts_string_int(self, handler):
        """Test conversion of string to int."""
        error = TypeError("expected int")
        result = handler.handle(error, "read", {"limit": "100"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["limit"] == 100

    def test_handle_converts_string_float(self, handler):
        """Test conversion of string to float."""
        error = TypeError("expected float")
        result = handler.handle(error, "search", {"threshold": "0.5"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.modified_args["threshold"] == 0.5

    def test_handle_skips_unconvertible(self, handler):
        """Test that unconvertible types result in skip."""
        error = TypeError("expected dict")
        result = handler.handle(error, "search", {"config": "not_a_dict"})

        assert result.action == RecoveryAction.SKIP


class TestChainOfResponsibility:
    """Tests for the chain of responsibility pattern."""

    @pytest.fixture
    def chain(self):
        return build_recovery_chain()

    def test_chain_handles_missing_parameter(self, chain):
        """Test chain handles missing parameter errors."""
        error = Exception("missing 1 required positional argument: 'path'")
        result = chain.process(error, "ls", {})

        assert result.action == RecoveryAction.RETRY_WITH_DEFAULTS
        assert result.metadata.get("handler") == "MissingParameterHandler"

    def test_chain_handles_tool_not_found(self, chain):
        """Test chain handles tool not found errors."""
        error = Exception("Tool 'symbol' not found")
        result = chain.process(error, "symbol", {})

        assert result.action == RecoveryAction.FALLBACK_TOOL
        assert result.metadata.get("handler") == "ToolNotFoundHandler"

    def test_chain_handles_network_error(self, chain):
        """Test chain handles network errors."""
        error = Exception("Connection timeout")
        result = chain.process(error, "web_fetch", {})

        assert result.action == RecoveryAction.RETRY
        assert result.metadata.get("handler") == "NetworkErrorHandler"

    def test_chain_handles_resource_budget_timeout_before_generic_network(self, chain):
        """Budget timeouts should produce local timeout guidance, not permission/network recovery."""
        error = Exception("graph mode 'overview' exceeded 90s budget")
        result = chain.process(error, "graph", {"mode": "overview"})

        assert result.action == RecoveryAction.SKIP
        assert result.metadata.get("handler") == "ResourceBudgetTimeoutHandler"
        assert result.metadata.get("error_kind") == "resource_budget_timeout"

    def test_chain_handles_file_not_found(self, chain):
        """Test chain handles file not found errors."""
        error = FileNotFoundError("test.py")
        result = chain.process(error, "read", {"path": "./test.py"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.metadata.get("handler") == "FileNotFoundHandler"

    def test_chain_handles_rate_limit(self, chain):
        """Test chain handles rate limit errors."""
        error = Exception("Rate limit exceeded")
        result = chain.process(error, "web_search", {})

        assert result.action == RecoveryAction.RETRY
        assert result.metadata.get("handler") == "RateLimitHandler"

    def test_chain_handles_permission_error(self, chain):
        """Test chain handles permission errors."""
        error = PermissionError("Permission denied")
        result = chain.process(error, "write", {})

        assert result.action == RecoveryAction.ASK_USER
        assert result.metadata.get("handler") == "PermissionErrorHandler"

    def test_chain_handles_type_error(self, chain):
        """Test chain handles type errors."""
        error = TypeError("expected int")
        result = chain.process(error, "read", {"limit": "100"})

        assert result.action == RecoveryAction.RETRY_WITH_INFERRED
        assert result.metadata.get("handler") == "TypeErrorHandler"

    def test_chain_aborts_unhandled_error(self, chain):
        """Test chain aborts on unhandled errors."""
        # Use a message that doesn't match any handler patterns
        error = Exception("Widget malfunction in flux capacitor")
        result = chain.process(error, "custom_tool", {})

        assert result.action == RecoveryAction.ABORT

    def test_chain_priority_missing_param_before_type(self, chain):
        """Test that missing parameter handler has priority over type handler."""
        # This error contains both 'type' and 'missing argument'
        error = Exception("missing 1 required positional argument: 'path'")
        result = chain.process(error, "ls", {})

        # Should be handled by MissingParameterHandler, not TypeErrorHandler
        assert result.metadata.get("handler") == "MissingParameterHandler"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_recovery_chain_returns_singleton(self):
        """Test that get_recovery_chain returns the same instance."""
        chain1 = get_recovery_chain()
        chain2 = get_recovery_chain()
        assert chain1 is chain2

    def test_recover_from_error_uses_default_chain(self):
        """Test that recover_from_error uses the default chain."""
        error = Exception("missing 1 required positional argument: 'path'")
        result = recover_from_error(error, "ls", {})

        assert result.action == RecoveryAction.RETRY_WITH_DEFAULTS
        assert result.modified_args["path"] == "."


class TestDependencyErrorFallback:
    """Tests for dependency/runtime error handling in ToolNotFoundHandler."""

    @pytest.fixture
    def handler(self):
        return ToolNotFoundHandler()

    def test_dependencies_missing_triggers_handler(self, handler):
        """Test that 'dependencies missing' error triggers ToolNotFoundHandler."""
        error = Exception("Tool 'semantic_code_search': dependencies missing for victor-coding")
        assert handler.can_handle(error, "semantic_code_search", {}) is True

    def test_no_module_named_triggers_handler(self, handler):
        """Test that 'No module named' error triggers ToolNotFoundHandler."""
        error = Exception("No module named 'victor.coding'")
        assert handler.can_handle(error, "semantic_code_search", {}) is True

    def test_import_error_triggers_handler(self, handler):
        """Test that ImportError triggers ToolNotFoundHandler."""
        error = ImportError("cannot import name 'SemanticSearch' from 'victor.coding'")
        assert handler.can_handle(error, "semantic_code_search", {}) is True

    def test_module_not_found_error_triggers_handler(self, handler):
        """Test that ModuleNotFoundError triggers ToolNotFoundHandler."""
        error = ModuleNotFoundError("No module named 'victor.coding.search'")
        assert handler.can_handle(error, "semantic_code_search", {}) is True

    def test_not_installed_triggers_handler(self, handler):
        """Test that 'not installed' error triggers ToolNotFoundHandler."""
        error = Exception("Package victor-coding is not installed")
        assert handler.can_handle(error, "code_search", {}) is True

    def test_code_search_falls_back_to_grep(self, handler):
        """Test that code_search falls back to grep."""
        error = Exception("Tool 'code_search' dependencies missing")
        result = handler.handle(error, "code_search", {"query": "MyClass"})

        assert result.action == RecoveryAction.FALLBACK_TOOL
        assert result.fallback_tool == "grep"

    def test_semantic_code_search_falls_back_to_code_search(self, handler):
        """Test that semantic_code_search falls back to code_search."""
        error = ImportError("No module named 'victor.coding'")
        result = handler.handle(error, "semantic_code_search", {"query": "MyClass"})

        assert result.action == RecoveryAction.FALLBACK_TOOL
        assert result.fallback_tool == "code_search"

    def test_chain_handles_import_error(self):
        """Test full chain handles ImportError via ToolNotFoundHandler."""
        chain = build_recovery_chain()
        error = ImportError("No module named 'victor.coding.search'")
        result = chain.process(error, "semantic_code_search", {"query": "test"})

        assert result.action == RecoveryAction.FALLBACK_TOOL
        assert result.fallback_tool == "code_search"
        assert result.metadata.get("handler") == "ToolNotFoundHandler"


class TestDeepSeekScenario:
    """Tests simulating the DeepSeek error from gap analysis."""

    def test_symbol_missing_file_path_recovery(self):
        """Test recovery from DeepSeek's symbol() missing file_path error."""
        chain = build_recovery_chain()
        error = Exception("symbol() missing 1 required positional argument: 'file_path'")
        result = chain.process(error, "symbol", {"symbol_name": "SearchResult"})

        assert result.action == RecoveryAction.RETRY_WITH_DEFAULTS
        assert "file_path" in result.modified_args
        assert result.modified_args["file_path"] == "."
        assert result.modified_args["symbol_name"] == "SearchResult"

    def test_cascading_recovery_attempts(self):
        """Test that we can handle cascading failures."""
        chain = build_recovery_chain()

        # First error: missing parameter
        error1 = Exception("missing 1 required positional argument: 'file_path'")
        result1 = chain.process(error1, "symbol", {"symbol_name": "SearchResult"})
        assert result1.should_retry is True

        # Second error: file not found (after retry with default path)
        error2 = FileNotFoundError("Symbol not found in .")
        result2 = chain.process(error2, "symbol", result1.modified_args)
        # Should try path variations or skip
        assert result2.action in (
            RecoveryAction.RETRY_WITH_INFERRED,
            RecoveryAction.SKIP,
        )

        # Third error: tool not found (after all retries)
        error3 = Exception("Tool 'symbol' not found")
        result3 = chain.process(error3, "symbol", {})
        assert result3.action == RecoveryAction.FALLBACK_TOOL
        assert result3.fallback_tool == "grep"
