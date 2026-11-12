# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Guard tests for slash-command name collisions (audit B3).

Discovery imports ``slash/commands/`` modules alphabetically and the later
module used to silently overwrite the earlier one at registry.register —
dead-commanding the loser while help, palette, and completions still
advertised it. Three real collisions existed: /status, /resume, /review.
"""

from __future__ import annotations

import logging

import pytest

from victor.ui.slash.registry import CommandRegistry, get_command_registry


def _discovered_registry() -> CommandRegistry:
    registry = get_command_registry()
    registry._ensure_discovered()
    return registry


class TestNoNameCollisions:
    def test_full_discovery_has_no_collisions(self):
        """The shipped command set must register without any name shadowing."""
        _discovered_registry().assert_no_collisions()

    def test_canonical_survivors_of_the_former_collisions(self):
        """Pin the commands that are SUPPOSED to own each contested name."""
        registry = _discovered_registry()
        assert registry.get("status").__class__.__module__.endswith("system")
        assert registry.get("resume").__class__.__module__.endswith("session")
        assert registry.get("review").__class__.__module__.endswith("tools")

    def test_renamed_plan_commands_are_reachable(self):
        """The former losers are reachable under their renamed identities."""
        registry = _discovered_registry()
        assert registry.get("plan-status") is not None
        assert registry.get("plan-resume") is not None
        assert registry.get("plan-status").__class__.__module__.endswith("status")
        assert registry.get("plan-resume").__class__.__module__.endswith("resume")


class TestCollisionDetection:
    def test_duplicate_registration_is_recorded_and_warned(self, caplog):
        registry = CommandRegistry()

        from victor.ui.slash.protocol import CommandMetadata

        class CmdFirst:
            @property
            def metadata(self) -> CommandMetadata:
                return CommandMetadata(
                    name="dup", description="first", usage="/dup", category="test"
                )

        class CmdSecond:
            @property
            def metadata(self) -> CommandMetadata:
                return CommandMetadata(
                    name="dup", description="second", usage="/dup", category="test"
                )

        with caplog.at_level(logging.WARNING):
            registry.register(CmdFirst())
            registry.register(CmdSecond())
        assert registry._collisions.get("dup") == CmdSecond.__module__
        assert any(
            r.levelno == logging.WARNING and "already registered" in r.message
            for r in caplog.records
        )
        with pytest.raises(AssertionError, match="/dup"):
            registry.assert_no_collisions()

    def test_clean_registration_leaves_no_collisions(self):
        from victor.ui.slash.protocol import CommandMetadata

        class Cmd:
            @property
            def metadata(self) -> CommandMetadata:
                return CommandMetadata(name="solo", description="", usage="/solo", category="test")

        registry = CommandRegistry()
        registry.register(Cmd())
        registry.assert_no_collisions()
        assert registry.get("solo") is not None


class TestRegisterClassTypeGuard:
    def test_register_class_rejects_non_class(self):
        """The stray @register_command on a helper function used to raise a
        swallowed TypeError on every import of import_export.py."""
        registry = CommandRegistry()
        with pytest.raises(TypeError, match="expects a command class"):
            registry.register_class(lambda ctx: None)  # type: ignore[arg-type]
