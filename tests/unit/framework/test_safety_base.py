"""Tests for base safety patterns (WS-5)."""

from victor.framework.safety.base_patterns import (
    BaseSafetyExtension,
    COMMON_PATTERNS,
)
from victor.security.safety.types import SafetyPattern


class TestBaseSafetyExtension:

    def test_base_returns_common_patterns(self):
        ext = BaseSafetyExtension()
        patterns = ext.get_patterns()
        assert len(patterns) == len(COMMON_PATTERNS)

    def test_get_bash_patterns_matches_get_patterns(self):
        ext = BaseSafetyExtension()
        assert ext.get_bash_patterns() == ext.get_patterns()

    def test_subclass_extends_patterns(self):
        class CustomSafety(BaseSafetyExtension):
            def get_additional_patterns(self):
                return [SafetyPattern(pattern=r"custom_cmd", description="Custom")]

        ext = CustomSafety()
        patterns = ext.get_patterns()
        assert len(patterns) == len(COMMON_PATTERNS) + 1
        assert any(p.description == "Custom" for p in patterns)

    def test_pattern_has_required_fields(self):
        for p in COMMON_PATTERNS:
            assert p.pattern
            assert p.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            assert p.category

    def test_pattern_is_canonical_type(self):
        for p in COMMON_PATTERNS:
            assert isinstance(p, SafetyPattern)

    def test_base_category(self):
        ext = BaseSafetyExtension()
        assert ext.get_category() == "general"
