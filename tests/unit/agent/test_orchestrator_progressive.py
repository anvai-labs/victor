"""Mirror tests for victor.agent.orchestrator_progressive.

Pins the progressive-tools block moved verbatim out of
victor/agent/orchestrator.py (facade decomposition): detection behavior,
the PROGRESSIVE_TOOLS proxy surface, import-time registration, and the
re-export identity the historical orchestrator module must preserve.
"""

import victor.agent.orchestrator as orchestrator_module
import victor.agent.orchestrator_progressive as progressive_module
from victor.agent.orchestrator_progressive import (
    PROGRESSIVE_TOOLS,
    _detect_mentioned_tools,
    _register_progressive_tools,
)
from victor.tools.progressive_registry import get_progressive_registry


class TestDetectMentionedTools:
    """_detect_mentioned_tools behavior pins (positive + negative)."""

    def test_call_pattern_detects_tool(self):
        assert _detect_mentioned_tools("Let me call read to check the file") == ["read"]

    def test_use_pattern_detects_tool(self):
        assert _detect_mentioned_tools("I'll use grep to search the codebase") == ["grep"]

    def test_parens_pattern_detects_tool(self):
        assert _detect_mentioned_tools("Calling read() to check contents") == ["read"]

    def test_the_tool_pattern_detects_tool(self):
        assert _detect_mentioned_tools("use the ls tool here") == ["ls"]

    def test_plain_text_detects_nothing(self):
        # Negative: prose without a tool-mention pattern yields no mentions.
        assert _detect_mentioned_tools("Just some regular text") == []

    def test_empty_text_detects_nothing(self):
        assert _detect_mentioned_tools("") == []

    def test_alias_resolves_to_canonical(self):
        # Negative-ish pin: a legacy alias mention resolves to its canonical name.
        mentions = _detect_mentioned_tools("let me call web_search now")
        assert "web_search" in mentions


class TestProgressiveToolsProxy:
    """PROGRESSIVE_TOOLS registry-proxy surface pins."""

    def test_getitem_known_tool(self):
        assert "path" in PROGRESSIVE_TOOLS["read"]

    def test_getitem_unknown_tool_raises_keyerror(self):
        # Negative: unknown tool is a KeyError, not a silent default.
        try:
            PROGRESSIVE_TOOLS["definitely_not_a_tool_xyz"]
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for unknown tool")

    def test_contains(self):
        assert "read" in PROGRESSIVE_TOOLS
        assert "definitely_not_a_tool_xyz" not in PROGRESSIVE_TOOLS

    def test_get_with_and_without_default(self):
        assert PROGRESSIVE_TOOLS.get("read") is not None
        assert PROGRESSIVE_TOOLS.get("nope_xyz") is None
        assert PROGRESSIVE_TOOLS.get("nope_xyz", ["d"]) == ["d"]

    def test_keys_and_iter(self):
        keys = set(PROGRESSIVE_TOOLS.keys())
        assert "read" in keys
        assert set(iter(PROGRESSIVE_TOOLS)) == keys

    def test_items_yields_lists(self):
        items = dict(PROGRESSIVE_TOOLS.items())
        assert isinstance(items["read"], list)


class TestRegistrationTiming:
    """Module-load registration + idempotency pins."""

    def test_registered_at_import_time(self):
        # The orchestrator module import (which imports the progressive module
        # at the top of its import block) must have registered the tools.
        registry = get_progressive_registry()
        assert registry.is_progressive("read")
        assert registry.is_progressive("grep")

    def test_registration_is_idempotent(self):
        _register_progressive_tools()
        _register_progressive_tools()
        registry = get_progressive_registry()
        assert registry.is_progressive("read")


class TestReExportSurface:
    """The orchestrator module must keep the historical attribute surface."""

    def test_detect_mentioned_tools_identity(self):
        # Positive: same function object, not a copy.
        assert orchestrator_module._detect_mentioned_tools is (
            progressive_module._detect_mentioned_tools
        )

    def test_progressive_tools_identity(self):
        assert orchestrator_module.PROGRESSIVE_TOOLS is progressive_module.PROGRESSIVE_TOOLS

    def test_names_in_all(self):
        # Negative: dropping the re-export from __all__ fails here.
        assert "_detect_mentioned_tools" in orchestrator_module.__all__
        assert "PROGRESSIVE_TOOLS" in orchestrator_module.__all__
