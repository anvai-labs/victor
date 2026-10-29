"""TDD tests for crash containment in vertical hook calls."""

import logging

from victor.core.verticals.base import VerticalBase


def _make_crashing_vertical(**overrides):
    """Create a vertical subclass where specified methods crash."""
    attrs = {
        "name": "crashy",
        "description": "crashy vertical",
        "get_name": classmethod(lambda cls: "crashy"),
        "get_description": classmethod(lambda cls: "crashy"),
        "get_tools": classmethod(lambda cls: ["read"]),
        "get_system_prompt": classmethod(lambda cls: "You are a test."),
    }
    for method_name, exc in overrides.items():

        def make_raiser(e):
            def raiser(cls):
                raise e

            return classmethod(raiser)

        attrs[method_name] = make_raiser(exc)

    return type("CrashyVertical", (VerticalBase,), attrs)


class TestGetConfigCrashContainment:

    def test_survives_get_tools_crash(self):
        V = _make_crashing_vertical(get_tools=RuntimeError("tools broke"))
        config = V.get_config(use_cache=False)
        assert config is not None
        # Should have empty/default tools
        assert config.name == "crashy"

    def test_survives_get_system_prompt_crash(self):
        V = _make_crashing_vertical(get_system_prompt=RuntimeError("prompt broke"))
        config = V.get_config(use_cache=False)
        assert config is not None
        assert config.system_prompt == ""

    def test_survives_get_stages_crash(self):
        V = _make_crashing_vertical(get_stages=RuntimeError("stages broke"))
        config = V.get_config(use_cache=False)
        assert config is not None

    def test_logs_warning_on_crash(self, caplog):
        V = _make_crashing_vertical(get_tools=RuntimeError("tools broke"))
        with caplog.at_level(logging.WARNING):
            V.get_config(use_cache=False)
        assert any("tools broke" in r.message or "get_tools" in r.message for r in caplog.records)

    def test_normal_vertical_unaffected(self):
        """Non-crashing verticals should work exactly as before."""

        class GoodVertical(VerticalBase):
            name = "good"
            description = "good vertical"

            @classmethod
            def get_name(cls):
                return "good"

            @classmethod
            def get_description(cls):
                return "good"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "You are good."

        config = GoodVertical.get_config(use_cache=False)
        assert config.name == "good"
        assert config.system_prompt == "You are good."
        assert "read" in str(config)
