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

"""Tests for AgentFactory profile-context window application.

A profile's ``context.max_context_tokens`` must reach the factory Settings
so ``resolve_context_window`` honors it (``source=settings``). Historically
the group survived only as ``__pydantic_extra__`` and never reached the
resolver — the factory is the single apply point (``SessionConfig.
apply_to_settings`` never touches ``context``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.config.context_settings import ContextSettings
from victor.config.profile_settings import ProfileConfig
from victor.framework.agent_factory import AgentFactory


def _profile(context_tokens=None):
    return ProfileConfig(
        provider="openai",
        model="qwen3.7-plus",
        **({"context": {"max_context_tokens": context_tokens}} if context_tokens else {}),
    )


def _factory(settings, profile_name="qwen-codingplan-direct", profiles=None):
    factory = AgentFactory.__new__(AgentFactory)
    factory._settings = settings
    factory._profile = profile_name
    factory._profiles_override = profiles
    return factory


class _Settings(SimpleNamespace):
    """Minimal settings stand-in exposing load_profiles + nested context."""

    def __init__(self, profiles, context=None):
        super().__init__(context=context or ContextSettings())
        self._profiles = profiles

    def load_profiles(self):
        return self._profiles


class TestApplyProfileContextWindow:
    def test_profile_window_applied_and_fields_set(self):
        settings = _Settings({"qwen-codingplan-direct": _profile(262144)})
        factory = _factory(settings)
        factory._apply_profile_context_window()
        assert settings.context.max_context_tokens == 262144
        assert "max_context_tokens" in settings.context.model_fields_set

    def test_profile_only_run_applies(self):
        """The exact `victor chat -p <profile>` case: no SDK overrides at all."""
        settings = _Settings({"p": _profile(199999)})
        factory = _factory(settings, profile_name="p")
        factory._apply_profile_context_window()
        assert settings.context.max_context_tokens == 199999

    def test_user_set_env_or_cli_value_wins(self):
        settings = _Settings(
            {"p": _profile(262144)},
            context=ContextSettings(max_context_tokens=9000),
        )
        factory = _factory(settings, profile_name="p")
        factory._apply_profile_context_window()
        assert settings.context.max_context_tokens == 9000

    def test_no_context_group_is_noop(self):
        settings = _Settings({"p": _profile()})
        before = settings.context
        factory = _factory(settings, profile_name="p")
        factory._apply_profile_context_window()
        assert settings.context is before

    def test_missing_profile_is_silent_noop(self):
        settings = _Settings({"other": _profile(262144)})
        before = settings.context
        factory = _factory(settings, profile_name="missing")
        factory._apply_profile_context_window()
        assert settings.context is before

    def test_broken_settings_never_raises(self):
        factory = _factory(MagicMock())  # load_profiles returns MagicMock
        factory._apply_profile_context_window()  # must not raise

    def test_only_user_set_fields_marked_after_merge(self):
        """Merging must not mark every nested default as explicitly set."""
        settings = _Settings(
            {"p": _profile(262144)},
            context=ContextSettings(),
        )
        factory = _factory(settings, profile_name="p")
        factory._apply_profile_context_window()
        fields = settings.context.model_fields_set
        assert "max_context_tokens" in fields
        # A default untouched by user or profile must not become "set".
        assert len(fields) <= 2  # at most: max_context_tokens (+ any future profile key)

    def test_resolver_sees_profile_window_as_settings_source(self):
        from victor.config.context_window import resolve_context_window

        settings = _Settings({"p": _profile(262144)})
        factory = _factory(settings, profile_name="p")
        factory._apply_profile_context_window()
        resolution = resolve_context_window("openai", "qwen3.7-plus", settings=settings)
        assert resolution.tokens == 262144
        assert resolution.source == "settings"

    def test_context_no_longer_leaks_into_provider_extras(self):
        """Typing `context` removes it from __pydantic_extra__ (provider kwargs)."""
        profile = _profile(262144)
        extras = profile.__pydantic_extra__ or {}
        assert "context" not in extras
