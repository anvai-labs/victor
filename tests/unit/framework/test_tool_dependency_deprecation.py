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

"""Unit tests for tool dependency deprecation helpers."""

import pytest
import warnings
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from victor.framework.tool_dependency_deprecation import (
    DeprecatedConstantDescriptor,
    VerticalDeprecationModule,
    create_vertical_deprecation_module,
    create_module_getattr,
)


class TestDeprecatedConstantDescriptor:
    """Tests for DeprecatedConstantDescriptor class."""

    def test_creates_descriptor_with_correct_attributes(self):
        """Test descriptor initialization stores attributes correctly."""

        def loader():
            return "test_value"

        descriptor = DeprecatedConstantDescriptor(
            constant_name="TEST_CONSTANT",
            deprecation_message="Use new_method() instead",
            loader=loader,
        )

        assert descriptor.constant_name == "TEST_CONSTANT"
        assert descriptor.deprecation_message == "Use new_method() instead"
        assert descriptor.loader is loader
        assert descriptor._loaded is False
        assert descriptor._warned is False
        assert descriptor._cached_value is None

    def test_get_emits_warning_on_first_access(self, caplog):
        """Test that deprecation log warning is emitted on first access."""

        def loader():
            return "test_value"

        descriptor = DeprecatedConstantDescriptor(
            constant_name="MY_CONSTANT",
            deprecation_message="MY_CONSTANT is deprecated",
            loader=loader,
        )

        class Container:
            const = descriptor

        _ = Container.const

        assert any("MY_CONSTANT is deprecated" in r.message for r in caplog.records)

    def test_get_only_warns_once(self, caplog):
        """Test that warning is only emitted once."""

        def loader():
            return "test_value"

        descriptor = DeprecatedConstantDescriptor(
            constant_name="MY_CONSTANT",
            deprecation_message="MY_CONSTANT is deprecated",
            loader=loader,
        )

        class Container:
            const = descriptor

        _ = Container.const
        _ = Container.const
        _ = Container.const

        dep_logs = [r for r in caplog.records if "MY_CONSTANT is deprecated" in r.message]
        assert len(dep_logs) == 1

    def test_get_caches_loaded_value(self):
        """Test that value is cached after first load."""
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return {"key": "value"}

        descriptor = DeprecatedConstantDescriptor(
            constant_name="MY_CONSTANT",
            deprecation_message="deprecated",
            loader=loader,
        )

        class Container:
            const = descriptor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Access multiple times
            val1 = Container.const
            val2 = Container.const
            val3 = Container.const

            # Loader should only be called once
            assert call_count == 1
            # All values should be the same cached object
            assert val1 is val2 is val3

    def test_get_returns_loaded_value(self):
        """Test that get returns the loaded value."""
        expected = {"dependencies": ["tool1", "tool2"]}

        def loader():
            return expected

        descriptor = DeprecatedConstantDescriptor(
            constant_name="MY_CONSTANT",
            deprecation_message="deprecated",
            loader=loader,
        )

        class Container:
            const = descriptor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = Container.const

            assert result == expected

    def test_reset_clears_cache_and_warning_state(self, caplog):
        """Test that reset clears cached value and warning state."""
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        # Use a unique message to avoid conflicts with other deprecation warnings
        unique_msg = "TEST_CONSTANT_DEPRECATED_xyz123"

        descriptor = DeprecatedConstantDescriptor(
            constant_name="MY_CONSTANT",
            deprecation_message=unique_msg,
            loader=loader,
        )

        class Container:
            const = descriptor

        # First access
        val1 = Container.const
        assert val1 == "value_1"
        first_count = len([r for r in caplog.records if unique_msg in r.message])
        assert first_count == 1

        # Reset
        descriptor.reset()

        # Access again should reload and warn again
        val2 = Container.const
        assert val2 == "value_2"
        second_count = len([r for r in caplog.records if unique_msg in r.message])
        assert second_count == 2

    def test_loader_exception_propagates(self):
        """Test that loader exceptions propagate correctly."""

        def failing_loader():
            raise ValueError("Load failed")

        descriptor = DeprecatedConstantDescriptor(
            constant_name="MY_CONSTANT",
            deprecation_message="deprecated",
            loader=failing_loader,
        )

        class Container:
            const = descriptor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with pytest.raises(ValueError, match="Load failed"):
                _ = Container.const


class TestVerticalDeprecationModule:
    """Tests for VerticalDeprecationModule class."""

    def test_init_stores_attributes(self):
        """Test that initialization stores attributes correctly."""
        yaml_path = Path("/path/to/config.yaml")

        module = VerticalDeprecationModule(
            vertical_name="coding",
            yaml_path=yaml_path,
            constant_prefix="CODING",
        )

        assert module.vertical_name == "coding"
        assert module.yaml_path == yaml_path
        assert module.constant_prefix == "CODING"

    def test_add_descriptor_creates_correct_constant_name(self):
        """Test that add_descriptor creates properly named constants."""
        module = VerticalDeprecationModule(
            vertical_name="coding",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="CODING",
        )

        module.add_descriptor(
            suffix="TOOL_DEPENDENCIES",
            extractor="dependencies",
            provider_method="get_dependencies",
        )

        constant_names = module.get_constant_names()
        assert "CODING_TOOL_DEPENDENCIES" in constant_names

    def test_getattr_returns_value_for_registered_constant(self):
        """Test that __getattr__ returns values for registered constants."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        # Create a mock descriptor
        mock_descriptor = MagicMock()
        mock_descriptor.__get__ = MagicMock(return_value="test_value")
        module._descriptors["TEST_CONSTANT"] = mock_descriptor

        result = module.TEST_CONSTANT

        assert result == "test_value"

    def test_getattr_raises_for_unknown_constant(self):
        """Test that __getattr__ raises AttributeError for unknown constants."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        with pytest.raises(AttributeError):
            _ = module.UNKNOWN_CONSTANT

    def test_getattr_raises_for_private_attributes(self):
        """Test that __getattr__ raises for private attributes."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        with pytest.raises(AttributeError):
            _ = module._private

    def test_get_constant_names_returns_all_registered(self):
        """Test that get_constant_names returns all registered names."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        module._descriptors["TEST_A"] = MagicMock()
        module._descriptors["TEST_B"] = MagicMock()
        module._descriptors["TEST_C"] = MagicMock()

        names = module.get_constant_names()

        assert names == {"TEST_A", "TEST_B", "TEST_C"}

    def test_reset_all_clears_config_cache_and_descriptors(self):
        """Test that reset_all clears everything."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        module._config_cache = {"cached": "data"}

        mock_descriptor = MagicMock()
        module._descriptors["TEST_CONST"] = mock_descriptor

        module.reset_all()

        assert module._config_cache is None
        mock_descriptor.reset.assert_called_once()


class TestCreateVerticalDeprecationModule:
    """Tests for create_vertical_deprecation_module factory."""

    def test_creates_module_with_correct_attributes(self):
        """Test factory creates module with correct attributes."""
        module = create_vertical_deprecation_module(
            vertical_name="research",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="RESEARCH",
            include_standard=False,
        )

        assert module.vertical_name == "research"
        assert module.constant_prefix == "RESEARCH"

    def test_includes_standard_constants_by_default(self):
        """Test that standard constants are included by default."""
        module = create_vertical_deprecation_module(
            vertical_name="coding",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="CODING",
        )

        names = module.get_constant_names()

        assert "CODING_TOOL_DEPENDENCIES" in names
        assert "CODING_TOOL_TRANSITIONS" in names
        assert "CODING_TOOL_CLUSTERS" in names
        assert "CODING_TOOL_SEQUENCES" in names
        assert "CODING_REQUIRED_TOOLS" in names
        assert "CODING_OPTIONAL_TOOLS" in names

    def test_excludes_standard_when_disabled(self):
        """Test that standard constants can be excluded."""
        module = create_vertical_deprecation_module(
            vertical_name="coding",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="CODING",
            include_standard=False,
        )

        names = module.get_constant_names()

        assert len(names) == 0

    def test_includes_extra_mappings(self):
        """Test that extra mappings are included."""
        module = create_vertical_deprecation_module(
            vertical_name="custom",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="CUSTOM",
            include_standard=False,
            extra_mappings={
                "MY_EXTRA": ("my_field", "get_my_field"),
            },
        )

        names = module.get_constant_names()

        assert "CUSTOM_MY_EXTRA" in names


class TestCreateModuleGetattr:
    """Tests for create_module_getattr function."""

    def test_returns_callable(self):
        """Test that factory returns a callable."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        getattr_fn = create_module_getattr(module)

        assert callable(getattr_fn)

    def test_returned_function_accesses_registered_constants(self):
        """Test that returned function accesses registered constants."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        mock_descriptor = MagicMock()
        mock_descriptor.__get__ = MagicMock(return_value="test_value")
        module._descriptors["TEST_CONST"] = mock_descriptor

        getattr_fn = create_module_getattr(module)

        result = getattr_fn("TEST_CONST")
        assert result == "test_value"

    def test_returned_function_raises_for_unknown(self):
        """Test that returned function raises for unknown attributes."""
        module = VerticalDeprecationModule(
            vertical_name="test",
            yaml_path=Path("/path/to/config.yaml"),
            constant_prefix="TEST",
        )

        getattr_fn = create_module_getattr(module)

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr_fn("UNKNOWN")
