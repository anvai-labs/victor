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

"""Tests for VerticalBase class - LSP compliance and core functionality."""

import asyncio
import time

import pytest
from typing import List
from unittest.mock import patch, MagicMock
import threading

from victor.core.verticals.base import VerticalBase, VerticalConfig
from victor.core.verticals.extension_loader import (
    VerticalExtensionLoader,
    start_extension_loader_metrics_reporter,
    stop_extension_loader_metrics_reporter,
)


class ConcreteVertical(VerticalBase):
    """Concrete implementation of VerticalBase for testing."""

    name = "test_vertical"
    description = "A test vertical for unit testing"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read", "write", "grep"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "You are a test assistant."


class TestGetExtensionsLSPCompliance:
    """Test suite for get_extensions() LSP compliance.

    LSP (Liskov Substitution Principle) requires that:
    1. The return type must be consistent (never None if declared as VerticalExtensions)
    2. Subclasses must be substitutable for the base class
    3. All code paths must return valid objects
    """

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_get_extensions_never_returns_none(self):
        """Verify that get_extensions() NEVER returns None.

        This is the primary LSP compliance test. The method must return
        a valid VerticalExtensions object in all scenarios.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = ConcreteVertical.get_extensions(use_cache=False)

        # Primary assertion: must not be None
        assert extensions is not None, "get_extensions() must never return None"

        # Type assertion: must be VerticalExtensions
        assert isinstance(
            extensions, VerticalExtensions
        ), f"Expected VerticalExtensions, got {type(extensions)}"

    def test_get_extensions_returns_valid_object_on_exception(self):
        """Verify get_extensions() returns valid VerticalExtensions even on exceptions.

        This tests the exception handler path to ensure LSP compliance
        when extension getter methods fail.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        class FailingVertical(VerticalBase):
            """Vertical whose extension getter raises an exception."""

            name = "failing_vertical"
            description = "A vertical that fails during extension loading"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Failing prompt"

            @classmethod
            def get_safety_extension(cls):
                # Simulate an error during extension loading
                raise RuntimeError("Simulated extension loading failure")

        # Force cache bypass to trigger fresh extension loading
        FailingVertical.clear_config_cache(clear_all=True)

        extensions = FailingVertical.get_extensions(use_cache=False)

        # Must still return a valid object, not None (LSP compliance)
        assert extensions is not None, "get_extensions() returned None on exception - LSP violation"
        assert isinstance(
            extensions, VerticalExtensions
        ), f"Expected VerticalExtensions, got {type(extensions)}"

        # Verify the returned extensions have proper default values
        assert extensions.middleware == []
        assert extensions.safety_extensions == []
        assert extensions.prompt_contributors == []

    def test_get_extensions_has_proper_default_values(self):
        """Verify that returned VerticalExtensions has proper default values.

        All fields should have sensible defaults (empty lists, None for optionals).
        """
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = ConcreteVertical.get_extensions(use_cache=False)

        # Verify default values for list fields
        assert isinstance(extensions.middleware, list), "middleware should be a list"
        assert isinstance(extensions.safety_extensions, list), "safety_extensions should be a list"
        assert isinstance(
            extensions.prompt_contributors, list
        ), "prompt_contributors should be a list"

        # Verify optional fields can be None (this is allowed)
        # But the container itself must exist
        assert hasattr(extensions, "mode_config_provider")
        assert hasattr(extensions, "tool_dependency_provider")
        assert hasattr(extensions, "workflow_provider")
        assert hasattr(extensions, "service_provider")
        assert hasattr(extensions, "rl_config_provider")
        assert hasattr(extensions, "team_spec_provider")
        assert hasattr(extensions, "enrichment_strategy")

    def test_get_extensions_caching_works(self):
        """Verify that caching returns the same object."""
        # First call - should cache
        extensions1 = ConcreteVertical.get_extensions(use_cache=True)
        # Second call - should return cached
        extensions2 = ConcreteVertical.get_extensions(use_cache=True)

        assert extensions1 is extensions2, "Caching should return same object"

    def test_get_extensions_cache_bypass_creates_new_object(self):
        """Verify that use_cache=False creates a new object."""
        # First call
        extensions1 = ConcreteVertical.get_extensions(use_cache=True)
        # Second call with cache bypass
        extensions2 = ConcreteVertical.get_extensions(use_cache=False)

        # Should be different objects (though may have same content)
        assert extensions1 is not extensions2, "Cache bypass should create new object"

    def test_get_extensions_consistent_type_across_subclasses(self):
        """Verify LSP: subclasses return the same type as base class would."""
        from victor.core.verticals.protocols import VerticalExtensions

        class AnotherVertical(VerticalBase):
            name = "another_vertical"
            description = "Another test vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Another prompt"

        extensions1 = ConcreteVertical.get_extensions(use_cache=False)
        extensions2 = AnotherVertical.get_extensions(use_cache=False)

        # Both must return VerticalExtensions (LSP)
        assert type(extensions1) is type(
            extensions2
        ), "LSP violation: different subclasses return different types"
        assert isinstance(extensions1, VerticalExtensions)
        assert isinstance(extensions2, VerticalExtensions)


class TestGetExtensionsAsync:
    """Tests for async extension loading API."""

    def setup_method(self):
        ConcreteVertical.clear_config_cache(clear_all=True)
        VerticalExtensionLoader.reset_extension_loader_metrics()

    @pytest.mark.asyncio
    async def test_get_extensions_async_never_returns_none(self):
        """Async API should return VerticalExtensions, never None."""
        from victor.core.verticals.protocols import VerticalExtensions

        extensions = await ConcreteVertical.get_extensions_async(use_cache=False)
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)

    @pytest.mark.asyncio
    async def test_get_extensions_async_caching_works(self):
        """Async API should reuse cache when enabled."""
        first = await ConcreteVertical.get_extensions_async(use_cache=True)
        second = await ConcreteVertical.get_extensions_async(use_cache=True)
        assert first is second

    @pytest.mark.asyncio
    async def test_get_extensions_sync_and_async_share_cache(self):
        """Sync and async extension loaders should share the same cache entry."""
        sync_extensions = ConcreteVertical.get_extensions(use_cache=True)
        async_extensions = await ConcreteVertical.get_extensions_async(use_cache=True)
        assert sync_extensions is async_extensions

    @pytest.mark.asyncio
    async def test_get_extensions_async_honors_strict_mode(self):
        """Strict async mode should raise ExtensionLoadError on extension failure."""
        from victor.core.errors import ExtensionLoadError

        class StrictFailingVertical(VerticalBase):
            name = "strict_failing_vertical_async"
            description = "Strict failing async vertical"
            strict_extension_loading = True

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Strict failing prompt"

            @classmethod
            def get_safety_extension(cls):
                raise RuntimeError("simulated async strict failure")

        with pytest.raises(ExtensionLoadError) as exc_info:
            await StrictFailingVertical.get_extensions_async(use_cache=False)

        assert exc_info.value.extension_type == "safety"

    @pytest.mark.asyncio
    async def test_get_extensions_async_parallel_loading(self):
        """Async loader should support parallel extension loading."""

        class BarrierVertical(VerticalBase):
            name = "barrier_vertical_async"
            description = "Uses barrier to verify parallel extension loading"
            strict_extension_loading = False  # Don't require all extensions to exist
            _barrier = threading.Barrier(3)

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Barrier vertical prompt"

            @classmethod
            def get_middleware(cls):
                cls._barrier.wait(timeout=1.0)
                return []

            @classmethod
            def get_safety_extension(cls):
                cls._barrier.wait(timeout=1.0)
                return object()

            @classmethod
            def get_prompt_contributor(cls):
                cls._barrier.wait(timeout=1.0)
                return object()

        extensions = await BarrierVertical.get_extensions_async(use_cache=False)
        assert extensions is not None
        assert len(extensions.safety_extensions) >= 1
        assert len(extensions.prompt_contributors) >= 1

    @pytest.mark.asyncio
    async def test_get_extensions_async_reuses_shared_executor(self):
        """Async extension loading should reuse the shared executor."""
        await ConcreteVertical.get_extensions_async(use_cache=False)
        first_executor = VerticalExtensionLoader._extension_executor
        assert first_executor is not None

        await ConcreteVertical.get_extensions_async(use_cache=False)
        second_executor = VerticalExtensionLoader._extension_executor
        assert first_executor is second_executor

    @pytest.mark.asyncio
    async def test_get_extensions_async_updates_loader_metrics(self):
        """Async loader should expose submission/in-flight completion metrics."""
        VerticalExtensionLoader.reset_extension_loader_metrics()
        await ConcreteVertical.get_extensions_async(use_cache=False)

        metrics = VerticalExtensionLoader.get_extension_loader_metrics()
        assert metrics["submitted"] >= 11
        assert metrics["completed"] == metrics["submitted"]
        assert metrics["in_flight"] == 0
        assert metrics["max_in_flight"] >= 1
        assert metrics["queue_limit"] == VerticalExtensionLoader._extension_executor_queue_limit

    @pytest.mark.asyncio
    async def test_get_extensions_async_records_pressure_threshold_events(self):
        """Queue/in-flight saturation should increment pressure counters."""
        pm = VerticalExtensionLoader._pressure_monitor
        old_settings = {
            "warn_queue": pm.warn_queue_threshold,
            "error_queue": pm.error_queue_threshold,
            "warn_in_flight": pm.warn_in_flight_threshold,
            "error_in_flight": pm.error_in_flight_threshold,
            "cooldown": pm.pressure_cooldown_seconds,
            "emit_events": pm.emit_pressure_events,
        }
        try:
            VerticalExtensionLoader.configure_extension_loader_pressure(
                warn_queue_threshold=1,
                error_queue_threshold=2,
                warn_in_flight_threshold=1,
                error_in_flight_threshold=2,
                cooldown_seconds=0.0,
                emit_events=False,
            )
            VerticalExtensionLoader.reset_extension_loader_metrics()
            await ConcreteVertical.get_extensions_async(use_cache=False)

            metrics = VerticalExtensionLoader.get_extension_loader_metrics()
            assert metrics["pressure_warnings"] + metrics["pressure_errors"] >= 1
        finally:
            VerticalExtensionLoader.configure_extension_loader_pressure(
                warn_queue_threshold=old_settings["warn_queue"],
                error_queue_threshold=old_settings["error_queue"],
                warn_in_flight_threshold=old_settings["warn_in_flight"],
                error_in_flight_threshold=old_settings["error_in_flight"],
                cooldown_seconds=old_settings["cooldown"],
                emit_events=old_settings["emit_events"],
            )

    @pytest.mark.asyncio
    async def test_emit_extension_loader_metrics_event(self):
        """Metrics event helper should emit a payload to observability bus."""
        emitted = []

        class _Bus:
            async def emit(self, topic, data, source):
                emitted.append((topic, data, source))

        VerticalExtensionLoader.emit_extension_loader_metrics_event(event_bus=_Bus())
        await asyncio.sleep(0.05)

        assert len(emitted) >= 1
        topic, data, source = emitted[-1]
        assert topic == "vertical.extensions.loader.metrics"
        assert source == "VerticalExtensionLoader"
        assert "metrics" in data

    def test_extension_loader_metrics_reporter_start_stop(self):
        """Periodic reporter helper should start and stop cleanly."""
        reporter = start_extension_loader_metrics_reporter(interval_seconds=0.05)
        assert reporter.is_running is True
        stop_extension_loader_metrics_reporter(timeout=1.0)


class TestVerticalBaseConfig:
    """Tests for VerticalBase.get_config() method."""

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_get_config_returns_vertical_config(self):
        """Verify get_config returns a VerticalConfig object."""
        config = ConcreteVertical.get_config()

        assert isinstance(config, VerticalConfig)
        assert config.system_prompt == "You are a test assistant."

    def test_get_config_caching(self):
        """Verify config caching works correctly."""
        config1 = ConcreteVertical.get_config(use_cache=True)
        config2 = ConcreteVertical.get_config(use_cache=True)

        assert config1 is config2, "Cached configs should be same object"

    def test_get_config_cache_key_uses_class_namespace(self):
        """Same class names in different modules should not collide in config cache."""
        ConcreteVertical.clear_config_cache(clear_all=True)

        def _tools_one(cls):
            return ["read"]

        def _tools_two(cls):
            return ["write"]

        def _prompt_one(cls):
            return "Prompt One"

        def _prompt_two(cls):
            return "Prompt Two"

        vertical_one = type(
            "DuplicateVertical",
            (VerticalBase,),
            {
                "name": "dup_one",
                "description": "Duplicate class name vertical one",
                "get_tools": classmethod(_tools_one),
                "get_system_prompt": classmethod(_prompt_one),
            },
        )
        vertical_one.__module__ = "test.verticals.one"

        vertical_two = type(
            "DuplicateVertical",
            (VerticalBase,),
            {
                "name": "dup_two",
                "description": "Duplicate class name vertical two",
                "get_tools": classmethod(_tools_two),
                "get_system_prompt": classmethod(_prompt_two),
            },
        )
        vertical_two.__module__ = "test.verticals.two"

        config_one = vertical_one.get_config(use_cache=True)
        config_two = vertical_two.get_config(use_cache=True)

        assert config_one.system_prompt == "Prompt One"
        assert config_two.system_prompt == "Prompt Two"
        assert config_one is not config_two


class TestGetCachedExtension:
    """Tests for VerticalBase._get_cached_extension() helper method."""

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_caches_extension_on_first_call(self):
        """Verify that factory is called once and result is cached."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"created": call_count}

        result1 = ConcreteVertical._get_cached_extension("test_key", factory)
        result2 = ConcreteVertical._get_cached_extension("test_key", factory)

        assert call_count == 1, "Factory should be called only once"
        assert result1 is result2, "Should return same cached object"
        assert result1["created"] == 1

    def test_different_keys_cache_separately(self):
        """Verify that different keys maintain separate cache entries."""
        result1 = ConcreteVertical._get_cached_extension("key_a", lambda: "value_a")
        result2 = ConcreteVertical._get_cached_extension("key_b", lambda: "value_b")

        assert result1 == "value_a"
        assert result2 == "value_b"
        assert result1 != result2

    def test_different_verticals_cache_separately(self):
        """Verify that different vertical subclasses have isolated caches."""

        class AnotherVertical(VerticalBase):
            name = "another"
            description = "Another"

            @classmethod
            def get_tools(cls) -> List[str]:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return ""

        # Both use same key but should be cached separately
        result1 = ConcreteVertical._get_cached_extension("shared_key", lambda: "concrete")
        result2 = AnotherVertical._get_cached_extension("shared_key", lambda: "another")

        assert result1 == "concrete"
        assert result2 == "another"

    def test_clear_config_cache_clears_extensions(self):
        """Verify that clear_config_cache() also clears extension cache."""
        # Populate cache
        result1 = ConcreteVertical._get_cached_extension("test_key", lambda: "original")
        assert result1 == "original"

        # Clear cache
        ConcreteVertical.clear_config_cache()

        # Should call factory again
        result2 = ConcreteVertical._get_cached_extension("test_key", lambda: "new_value")
        assert result2 == "new_value"

    def test_same_class_name_different_modules_do_not_share_extension_cache(self):
        """Extension cache should be isolated by module + class, not class name only."""
        ConcreteVertical.clear_config_cache(clear_all=True)

        def _tools_one(cls):
            return ["read"]

        def _tools_two(cls):
            return ["write"]

        def _prompt_one(cls):
            return "Prompt One"

        def _prompt_two(cls):
            return "Prompt Two"

        vertical_one = type(
            "DuplicateVertical",
            (VerticalBase,),
            {
                "name": "dup_ext_one",
                "description": "Duplicate class name extension vertical one",
                "get_tools": classmethod(_tools_one),
                "get_system_prompt": classmethod(_prompt_one),
            },
        )
        vertical_one.__module__ = "test.verticals.extensions.one"

        vertical_two = type(
            "DuplicateVertical",
            (VerticalBase,),
            {
                "name": "dup_ext_two",
                "description": "Duplicate class name extension vertical two",
                "get_tools": classmethod(_tools_two),
                "get_system_prompt": classmethod(_prompt_two),
            },
        )
        vertical_two.__module__ = "test.verticals.extensions.two"

        first = vertical_one._get_cached_extension("shared_key", lambda: "one")
        second = vertical_two._get_cached_extension("shared_key", lambda: "two")

        assert first == "one"
        assert second == "two"


class TestDefaultStageDefinitions:
    """Tests for VerticalBase default stage definitions."""

    def test_has_seven_stages(self):
        """Verify default implementation has 7 stages."""
        stages = ConcreteVertical.get_stages()
        assert len(stages) == 7

    def test_required_stages_present(self):
        """Verify all required stages are defined."""
        stages = ConcreteVertical.get_stages()
        required = {
            "INITIAL",
            "PLANNING",
            "READING",
            "ANALYSIS",
            "EXECUTION",
            "VERIFICATION",
            "COMPLETION",
        }
        assert set(stages.keys()) == required

    def test_stages_have_comprehensive_keywords(self):
        """Verify stages have comprehensive keyword lists."""
        stages = ConcreteVertical.get_stages()

        # Each stage should have at least 8 keywords
        for stage_name, stage in stages.items():
            assert (
                len(stage.keywords) >= 8
            ), f"{stage_name} has too few keywords: {len(stage.keywords)}"

    def test_stages_have_valid_transitions(self):
        """Verify stage transitions form valid workflow graph."""
        stages = ConcreteVertical.get_stages()

        # All next_stages should reference valid stage names
        for stage_name, stage in stages.items():
            for next_stage in stage.next_stages:
                assert next_stage in stages, f"{stage_name} references invalid stage: {next_stage}"

        # COMPLETION should have no next stages (terminal state)
        assert len(stages["COMPLETION"].next_stages) == 0

        # INITIAL should be able to reach other stages
        assert len(stages["INITIAL"].next_stages) > 0

    def test_stages_have_descriptions(self):
        """Verify all stages have non-empty descriptions."""
        stages = ConcreteVertical.get_stages()

        for stage_name, stage in stages.items():
            assert stage.description, f"{stage_name} has empty description"
            assert len(stage.description) > 20, f"{stage_name} description too short"


class TestStrictExtensionLoading:
    """Tests for strict extension loading in VerticalBase.get_extensions().

    These tests verify the LSP-compliant error handling behavior where:
    - strict=True raises ExtensionLoadError on any failure
    - strict=False collects errors and returns partial extensions
    - required_extensions cause failures even in non-strict mode
    """

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_strict_mode_raises_on_any_error(self):
        """Verify strict=True returns valid extensions with lazy loading.

        With lazy extension loading, errors are deferred until access time
        and silently handled — the failed extension returns its default value.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        class StrictFailingVertical(VerticalBase):
            """Vertical that fails during extension loading."""

            name = "strict_failing"
            description = "A vertical that fails in strict mode"
            strict_extension_loading = True  # Enable strict mode at class level

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Strict failing prompt"

            @classmethod
            def get_safety_extension(cls):
                raise RuntimeError("Simulated safety extension failure")

        StrictFailingVertical.clear_config_cache(clear_all=True)

        # With lazy loading, get_extensions() always succeeds — errors are
        # deferred to property access where they are silently handled.
        extensions = StrictFailingVertical.get_extensions(use_cache=False)
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)

        # Accessing the failed extension returns the default (empty list)
        assert extensions.safety_extensions == []

    def test_strict_parameter_overrides_class_setting(self):
        """Verify strict parameter is accepted but lazy loading handles errors silently.

        With lazy extension loading, both strict=False and strict=True return
        valid extensions. Failed extensions resolve to their default values
        when accessed.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        class NonStrictVertical(VerticalBase):
            """Vertical with strict mode disabled at class level."""

            name = "non_strict"
            description = "A non-strict vertical"
            strict_extension_loading = False  # Disabled at class level

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Non-strict prompt"

            @classmethod
            def get_middleware(cls):
                raise RuntimeError("Middleware failure")

        NonStrictVertical.clear_config_cache(clear_all=True)

        # Non-strict: returns valid extensions, failed middleware resolves to []
        extensions = NonStrictVertical.get_extensions(use_cache=False)
        assert extensions is not None
        assert extensions.middleware == []

        # Clear cache and test with strict=True override
        NonStrictVertical.clear_config_cache(clear_all=True)

        # With lazy loading, strict=True also returns valid extensions —
        # errors are handled silently during lazy property access.
        extensions = NonStrictVertical.get_extensions(use_cache=False, strict=True)
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)
        assert extensions.middleware == []

    def test_required_extensions_fail_even_in_non_strict_mode(self):
        """Verify required_extensions are handled via lazy loading.

        With lazy extension loading, required_extensions failures are deferred
        to property access time and handled silently — the failed extension
        returns its default value.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        class RequiredExtVertical(VerticalBase):
            """Vertical with required extensions."""

            name = "required_ext"
            description = "A vertical with required extensions"
            strict_extension_loading = False  # Non-strict mode
            required_extensions = {"safety"}  # Safety is required

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Required ext prompt"

            @classmethod
            def get_safety_extension(cls):
                raise RuntimeError("Critical safety failure")

        RequiredExtVertical.clear_config_cache(clear_all=True)

        # With lazy loading, get_extensions() always returns valid extensions.
        # Even required extension failures are handled silently at access time.
        extensions = RequiredExtVertical.get_extensions(use_cache=False)
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)

        # Failed required extension resolves to empty list
        assert extensions.safety_extensions == []

    def test_non_strict_mode_returns_partial_extensions(self):
        """Verify non-strict mode returns partial extensions with failed components empty."""
        from victor.core.verticals.protocols import VerticalExtensions

        class PartialFailVertical(VerticalBase):
            """Vertical where some extensions fail."""

            name = "partial_fail"
            description = "A vertical with partial failures"
            strict_extension_loading = False

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Partial fail prompt"

            @classmethod
            def get_middleware(cls):
                # This succeeds
                return []

            @classmethod
            def get_safety_extension(cls):
                # This fails (non-required)
                raise RuntimeError("Safety failure")

            @classmethod
            def get_prompt_contributor(cls):
                # This succeeds
                return None

        PartialFailVertical.clear_config_cache(clear_all=True)

        extensions = PartialFailVertical.get_extensions(use_cache=False)

        # Should return valid extensions despite the failure
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)

        # Failed extension should have empty/None value
        assert extensions.safety_extensions == []

        # Successful extensions should work normally
        assert extensions.middleware == []

    def test_extension_load_error_has_proper_attributes(self):
        """Verify ExtensionLoadError has all expected attributes."""
        from victor.core.errors import ExtensionLoadError, ErrorCategory, ErrorSeverity

        original = RuntimeError("Original error")
        error = ExtensionLoadError(
            message="Test error",
            extension_type="safety",
            vertical_name="test_vertical",
            original_error=original,
            is_required=True,
        )

        assert error.extension_type == "safety"
        assert error.vertical_name == "test_vertical"
        assert error.original_error is original
        assert error.is_required is True
        assert error.category == ErrorCategory.CONFIG_INVALID
        assert error.severity == ErrorSeverity.CRITICAL  # Critical because is_required=True

        # Check details dict
        assert error.details["extension_type"] == "safety"
        assert error.details["vertical_name"] == "test_vertical"
        assert error.details["is_required"] is True
        assert error.details["original_error_type"] == "RuntimeError"

    def test_non_required_extension_error_has_warning_severity(self):
        """Verify non-required extension errors have WARNING severity."""
        from victor.core.errors import ExtensionLoadError, ErrorSeverity

        error = ExtensionLoadError(
            message="Test error",
            extension_type="enrichment",
            vertical_name="test_vertical",
            original_error=RuntimeError("Test"),
            is_required=False,
        )

        assert error.severity == ErrorSeverity.WARNING

    def test_backward_compatibility_default_non_strict(self):
        """Verify backward compatibility: default is non-strict mode."""
        # ConcreteVertical should have default strict_extension_loading=False
        assert ConcreteVertical.strict_extension_loading is False
        assert ConcreteVertical.required_extensions == set()

    def test_multiple_extension_failures_reports_first_critical(self):
        """Verify that multiple failures are handled silently with lazy loading.

        With lazy extension loading, all failures are deferred to property
        access time and handled silently — each failed extension returns
        its default value independently.
        """
        from victor.core.verticals.protocols import VerticalExtensions

        class MultiFailVertical(VerticalBase):
            """Vertical with multiple extension failures."""

            name = "multi_fail"
            description = "Multiple failures"
            strict_extension_loading = True  # All failures are critical

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Multi fail"

            @classmethod
            def get_middleware(cls):
                raise RuntimeError("Middleware failure")

            @classmethod
            def get_safety_extension(cls):
                raise RuntimeError("Safety failure")

        MultiFailVertical.clear_config_cache(clear_all=True)

        # With lazy loading, get_extensions() always succeeds
        extensions = MultiFailVertical.get_extensions(use_cache=False)
        assert extensions is not None
        assert isinstance(extensions, VerticalExtensions)

        # Each failed extension resolves to its default independently
        assert extensions.middleware == []
        assert extensions.safety_extensions == []


class TestIntegrationResultExtensionErrors:
    """Tests for IntegrationResult extension error tracking."""

    def test_add_extension_error_creates_error_info(self):
        """Verify add_extension_error creates proper ExtensionLoadErrorInfo."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        result.add_extension_error(
            extension_type="safety",
            vertical_name="test_vertical",
            error_message="Test error",
            is_required=False,
            original_exception=RuntimeError("Original"),
        )

        assert len(result.extension_errors) == 1
        error = result.extension_errors[0]
        assert error.extension_type == "safety"
        assert error.vertical_name == "test_vertical"
        assert error.error_message == "Test error"
        assert error.is_required is False
        assert error.original_exception_type == "RuntimeError"

    def test_required_extension_error_sets_failed_status(self):
        """Verify required extension errors set validation_status to failed."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        assert result.validation_status == "success"

        result.add_extension_error(
            extension_type="safety",
            vertical_name="test_vertical",
            error_message="Critical failure",
            is_required=True,
        )

        assert result.success is False
        assert result.validation_status == "failed"

    def test_non_required_extension_error_sets_partial_status(self):
        """Verify non-required extension errors set validation_status to partial."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        assert result.validation_status == "success"

        result.add_extension_error(
            extension_type="enrichment",
            vertical_name="test_vertical",
            error_message="Non-critical failure",
            is_required=False,
        )

        assert result.success is True  # Still success for non-required
        assert result.validation_status == "partial"

    def test_to_dict_includes_extension_errors(self):
        """Verify to_dict includes extension_errors and validation_status."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        result.add_extension_error(
            extension_type="safety",
            vertical_name="test_vertical",
            error_message="Test error",
            is_required=False,
        )

        data = result.to_dict()

        assert "validation_status" in data
        assert data["validation_status"] == "partial"
        assert "extension_errors" in data
        assert len(data["extension_errors"]) == 1
        assert data["extension_errors"][0]["extension_type"] == "safety"

    def test_from_dict_restores_extension_errors(self):
        """Verify from_dict properly restores extension_errors."""
        from victor.framework.vertical_integration import IntegrationResult

        original = IntegrationResult()
        original.add_extension_error(
            extension_type="middleware",
            vertical_name="test_vertical",
            error_message="Middleware error",
            is_required=True,
        )

        data = original.to_dict()
        restored = IntegrationResult.from_dict(data)

        assert len(restored.extension_errors) == 1
        assert restored.extension_errors[0].extension_type == "middleware"
        assert restored.extension_errors[0].is_required is True
        assert restored.validation_status == "failed"

    def test_has_extension_errors_method(self):
        """Verify has_extension_errors method works correctly."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        assert result.has_extension_errors() is False

        result.add_extension_error(
            extension_type="safety",
            vertical_name="test",
            error_message="Error",
        )
        assert result.has_extension_errors() is True

    def test_get_required_extension_failures_method(self):
        """Verify get_required_extension_failures filters correctly."""
        from victor.framework.vertical_integration import IntegrationResult

        result = IntegrationResult()
        result.add_extension_error(
            extension_type="safety",
            vertical_name="test",
            error_message="Required failure",
            is_required=True,
        )
        result.add_extension_error(
            extension_type="enrichment",
            vertical_name="test",
            error_message="Optional failure",
            is_required=False,
        )

        required = result.get_required_extension_failures()
        assert len(required) == 1
        assert required[0].extension_type == "safety"


class TestConfigCacheTTL:
    """Tests for config cache TTL expiry behavior."""

    def setup_method(self):
        """Clear caches before each test."""
        ConcreteVertical.clear_config_cache(clear_all=True)

    def test_cache_returns_within_ttl(self):
        """Verify same object reference is returned within TTL window."""
        config1 = ConcreteVertical.get_config(use_cache=True)
        config2 = ConcreteVertical.get_config(use_cache=True)

        assert config1 is config2, "Should return same cached object within TTL"

    def test_cache_expires_after_ttl(self):
        """Verify cache rebuilds after TTL expires."""
        # Set a very short TTL
        original_ttl = ConcreteVertical._config_cache_ttl
        ConcreteVertical._config_cache_ttl = 0.01  # 10ms

        try:
            config1 = ConcreteVertical.get_config(use_cache=True)
            time.sleep(0.02)  # Wait for TTL to expire
            config2 = ConcreteVertical.get_config(use_cache=True)

            assert config1 is not config2, "Should rebuild config after TTL expires"
        finally:
            ConcreteVertical._config_cache_ttl = original_ttl

    def test_custom_ttl_per_vertical(self):
        """Verify subclass TTL overrides affect cache expiry behavior."""

        class ShortTTLVertical(VerticalBase):
            name = "short_ttl"
            description = "Short TTL vertical"
            _config_cache_ttl = 0.01  # 10ms custom TTL

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Short TTL prompt"

        ShortTTLVertical.clear_config_cache(clear_all=True)
        config1 = ShortTTLVertical.get_config(use_cache=True)
        time.sleep(0.02)
        config2 = ShortTTLVertical.get_config(use_cache=True)

        assert config1 is not config2
        assert ShortTTLVertical._config_cache_ttl == 0.01
        assert ConcreteVertical._config_cache_ttl == 300.0

    def test_use_cache_false_bypasses_ttl(self):
        """Verify use_cache=False always rebuilds regardless of TTL."""
        config1 = ConcreteVertical.get_config(use_cache=True)
        config2 = ConcreteVertical.get_config(use_cache=False)

        assert config1 is not config2, "use_cache=False should always rebuild"
