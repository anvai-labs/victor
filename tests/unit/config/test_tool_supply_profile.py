# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the provider-economics tool-supply resolver (tool-supply P1)."""

from __future__ import annotations

from victor.config.tool_tiers import ToolSupplyProfile, resolve_tool_supply_profile
from victor.tools.enums import SchemaLevel


class _Provider:
    def __init__(self, *, prompt_cache=False, kv_cache=False, window=8192):
        self._pc = prompt_cache
        self._kv = kv_cache
        self._cw = window

    def supports_prompt_caching(self):
        return self._pc

    def supports_kv_prefix_caching(self):
        return self._kv

    def context_window(self, model=None):
        return self._cw


def test_prompt_caching_provider_full_uncapped():
    # Any window: caching makes the full set nearly free -> never cap or tier.
    for win in (8192, 200_000):
        p = resolve_tool_supply_profile(_Provider(prompt_cache=True, window=win))
        assert p == ToolSupplyProfile("none", "additive", SchemaLevel.FULL, None, None)


def test_kv_large_window_full_uncapped():
    p = resolve_tool_supply_profile(_Provider(kv_cache=True, window=64_000))
    assert p.cap_mode == "none"
    assert p.default_schema == SchemaLevel.FULL
    assert p.max_full_tools is None
    assert p.budget_tokens is None


def test_kv_mid_window_compact_stub_budget():
    p = resolve_tool_supply_profile(_Provider(kv_cache=True, window=24_000), fallback_max_tools=10)
    assert p.cap_mode == "stub"
    assert p.default_schema == SchemaLevel.COMPACT
    assert p.max_full_tools == 10
    assert p.budget_tokens == 6000  # 25% of 24k


def test_small_window_stub():
    p = resolve_tool_supply_profile(_Provider(kv_cache=True, window=8192), fallback_max_tools=8)
    assert p.cap_mode == "stub"
    assert p.default_schema == SchemaLevel.STUB
    assert p.max_full_tools == 8
    assert p.budget_tokens == 2048


def test_no_cache_large_window_tiers_compact_not_full():
    # No-cache provider pays full tokens every call -> tier even with a big window
    # (only kv/prompt caching earns the uncapped FULL set).
    p = resolve_tool_supply_profile(_Provider(window=128_000), fallback_max_tools=12)
    assert p.cap_mode == "stub"
    assert p.default_schema == SchemaLevel.COMPACT
    assert p.budget_tokens == 32_000


def test_no_cache_small_window_stub():
    p = resolve_tool_supply_profile(_Provider(window=8192))
    assert p.cap_mode == "stub"
    assert p.default_schema == SchemaLevel.STUB


def test_explicit_window_overrides_provider():
    # Provider reports 8k, but caller passes 64k -> kv large path.
    p = resolve_tool_supply_profile(_Provider(kv_cache=True, window=8192), context_window=64_000)
    assert p.default_schema == SchemaLevel.FULL
    assert p.cap_mode == "none"


def test_session_lock_always_additive():
    for prov in (
        _Provider(prompt_cache=True),
        _Provider(kv_cache=True, window=64_000),
        _Provider(kv_cache=True, window=24_000),
        _Provider(window=8192),
    ):
        assert resolve_tool_supply_profile(prov).session_lock == "additive"


def test_bare_provider_falls_back_conservatively():
    # Provider missing all capability methods -> most conservative profile, default window.
    class _Bare:
        pass

    p = resolve_tool_supply_profile(_Bare())
    assert p.cap_mode == "stub"
    assert p.default_schema == SchemaLevel.STUB
    assert p.budget_tokens == 2048  # 25% of the 8192 default window


def test_capability_probe_swallows_errors():
    class _Boom:
        def supports_prompt_caching(self):
            raise RuntimeError("boom")

        def supports_kv_prefix_caching(self):
            raise RuntimeError("boom")

        def context_window(self, model=None):
            raise RuntimeError("boom")

    # Must not raise; falls back to the conservative small-window profile.
    p = resolve_tool_supply_profile(_Boom())
    assert p.cap_mode == "stub"
    assert p.default_schema == SchemaLevel.STUB


def test_profile_is_frozen():
    import dataclasses

    import pytest

    p = resolve_tool_supply_profile(_Provider(prompt_cache=True))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.cap_mode = "hard"  # type: ignore[misc]
