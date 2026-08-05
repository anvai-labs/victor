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

"""Unit tests for workflow compiler plugin system.

Tests the simple plugin registry for third-party vertical extensibility.
"""

import pytest

from victor.workflows.compiler_registry import (
    register_compiler,
    unregister_compiler,
    get_compiler,
    is_registered,
    list_compilers,
)
from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin


class MockCompilerPlugin(WorkflowCompilerPlugin):
    """Mock plugin for testing."""

    def __init__(self, **options):
        self.options = options

    def compile(self, source, *, workflow_name=None, validate=True):
        # Return mock compiled object
        return {"source": source, "workflow_name": workflow_name}


@pytest.mark.unit
@pytest.mark.workflows
class TestCompilerRegistry:
    """Test compiler registry functionality."""

    def setup_method(self):
        """Clean registry before each test."""
        # Unregister any plugins from previous tests
        for name in list_compilers():
            try:
                unregister_compiler(name)
            except KeyError:
                pass

    def teardown_method(self):
        """Clean registry after each test."""
        # Unregister any mock plugins
        for name in ["mock1", "mock2", "mock", "test1"]:
            try:
                unregister_compiler(name)
            except KeyError:
                pass

    def test_register_plugin(self):
        """Test registering a plugin."""
        register_compiler("mock", MockCompilerPlugin)

        assert is_registered("mock")

    def test_register_multiple_plugins(self):
        """Test registering multiple plugins."""
        register_compiler("mock1", MockCompilerPlugin)
        register_compiler("mock2", MockCompilerPlugin)

        assert is_registered("mock1")
        assert is_registered("mock2")
        assert set(list_compilers()) == {"mock1", "mock2"}

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        register_compiler("mock", MockCompilerPlugin)

        assert is_registered("mock")

        unregister_compiler("mock")

        assert not is_registered("mock")

    def test_unregister_nonexistent_plugin(self):
        """Test unregistering nonexistent plugin raises KeyError."""
        with pytest.raises(KeyError, match="No compiler plugin registered"):
            unregister_compiler("nonexistent")

    def test_get_compiler(self):
        """Test getting a compiler instance."""
        register_compiler("mock", MockCompilerPlugin)

        compiler = get_compiler("mock", option1="value1")

        assert compiler is not None
        assert compiler.options == {"option1": "value1"}

    def test_get_compiler_unknown(self):
        """Test getting unknown compiler raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compiler plugin"):
            get_compiler("unknown")

    def test_list_compilers_empty(self):
        """Test listing compilers when none registered."""
        compilers = list_compilers()

        assert compilers == []

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered compiler."""
        assert not is_registered("yaml")


@pytest.mark.unit
@pytest.mark.workflows
class TestWorkflowCompilerPlugin:
    """Test WorkflowCompilerPlugin protocol."""

    def test_plugin_has_compile_method(self):
        """Test plugin has compile method."""
        plugin = MockCompilerPlugin()

        assert hasattr(plugin, "compile")
        assert callable(plugin.compile)

    def test_plugin_has_validate_source_method(self):
        """Test plugin has validate_source method."""
        plugin = MockCompilerPlugin()

        assert hasattr(plugin, "validate_source")
        assert callable(plugin.validate_source)

    def test_plugin_validate_source_default(self):
        """Test plugin validate_source returns True by default."""
        plugin = MockCompilerPlugin()

        assert plugin.validate_source("any source") is True

    def test_plugin_has_get_cache_key_method(self):
        """Test plugin has get_cache_key method."""
        plugin = MockCompilerPlugin()

        assert hasattr(plugin, "get_cache_key")
        assert callable(plugin.get_cache_key)

    def test_plugin_get_cache_key_default(self):
        """Test plugin get_cache_key returns source by default."""
        plugin = MockCompilerPlugin()

        assert plugin.get_cache_key("my-source") == "my-source"

    def test_plugin_compile_with_workflow_name(self):
        """Test plugin compile method accepts workflow_name parameter."""
        plugin = MockCompilerPlugin()

        result = plugin.compile("source.yaml", workflow_name="my_workflow")

        assert result["workflow_name"] == "my_workflow"

    def test_plugin_compile_with_validation_disabled(self):
        """Test plugin compile method accepts validate parameter."""
        plugin = MockCompilerPlugin()

        result = plugin.compile("source.yaml", validate=False)

        assert result["source"] == "source.yaml"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
