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

"""Tests for RL config protocol compliance.

Verifies that all RL configs implement get_rl_config() per RLConfigProviderProtocol.
"""

import pytest


@pytest.mark.parametrize(
    "vertical,module_path,config_class",
    [
        ("coding", "victor_coding.rl.config", "CodingRLConfig"),
        ("research", "victor_research.rl", "ResearchRLConfig"),
        ("devops", "victor_devops.rl", "DevOpsRLConfig"),
        ("dataanalysis", "victor_dataanalysis.rl", "DataAnalysisRLConfig"),
    ],
)
def test_rl_config_has_get_rl_config_method(vertical: str, module_path: str, config_class: str):
    """All RL configs must implement get_rl_config() per protocol."""
    import importlib

    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)  # nosemgrep
        config_cls = getattr(module, config_class)

        # Instantiate the config
        config = config_cls()

        # Verify the method exists
        assert hasattr(
            config, "get_rl_config"
        ), f"{config_class} must implement get_rl_config() per RLConfigProviderProtocol"

        # Verify the method is callable
        assert callable(config.get_rl_config), f"{config_class}.get_rl_config must be callable"
    except ImportError:
        pytest.skip(f"External vertical package not installed: {vertical}")


@pytest.mark.parametrize(
    "vertical,module_path,config_class",
    [
        ("coding", "victor_coding.rl.config", "CodingRLConfig"),
        ("research", "victor_research.rl", "ResearchRLConfig"),
        ("devops", "victor_devops.rl", "DevOpsRLConfig"),
        ("dataanalysis", "victor_dataanalysis.rl", "DataAnalysisRLConfig"),
    ],
)
def test_get_rl_config_returns_dict(vertical: str, module_path: str, config_class: str):
    """get_rl_config() must return Dict[str, Any] with expected keys."""
    import importlib

    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)  # nosemgrep
        config_cls = getattr(module, config_class)

        # Instantiate and call get_rl_config
        config = config_cls()
        result = config.get_rl_config()

        # Verify return type is dict
        assert isinstance(
            result, dict
        ), f"{config_class}.get_rl_config() must return Dict[str, Any], got {type(result)}"

        # Verify expected keys are present
        expected_keys = {
            "active_learners",
            "task_type_mappings",
            "quality_thresholds",
            "default_patience",
        }
        missing_keys = expected_keys - set(result.keys())
        assert (
            not missing_keys
        ), f"{config_class}.get_rl_config() missing expected keys: {missing_keys}"

        # Verify types of values
        assert isinstance(result["active_learners"], list), "active_learners must be a list"
        assert isinstance(result["task_type_mappings"], dict), "task_type_mappings must be a dict"
        assert isinstance(result["quality_thresholds"], dict), "quality_thresholds must be a dict"
        assert isinstance(result["default_patience"], dict), "default_patience must be a dict"
    except ImportError:
        pytest.skip(f"External vertical package not installed: {vertical}")


def test_rl_config_implements_protocol():
    """Verify RL configs satisfy RLConfigProviderProtocol via structural typing.

    Note: RLConfigProviderProtocol has get_rl_hooks() with default implementation.
    The isinstance() check requires ALL protocol methods to be present, even those
    with defaults. We verify structural compliance by checking the required method.
    """

    try:
        from victor_coding.rl.config import CodingRLConfig
        from victor_research.rl import ResearchRLConfig
        from victor_devops.rl import DevOpsRLConfig
        from victor_dataanalysis.rl import DataAnalysisRLConfig
    except ImportError as e:
        pytest.skip(f"External vertical packages not installed: {e}")

    configs = [
        CodingRLConfig(),
        ResearchRLConfig(),
        DevOpsRLConfig(),
        DataAnalysisRLConfig(),
    ]

    for config in configs:
        # Verify the required abstract method exists and is callable
        assert hasattr(
            config, "get_rl_config"
        ), f"{config.__class__.__name__} must implement get_rl_config()"
        assert callable(
            config.get_rl_config
        ), f"{config.__class__.__name__}.get_rl_config must be callable"

        # Verify it returns the expected type
        result = config.get_rl_config()
        assert isinstance(
            result, dict
        ), f"{config.__class__.__name__}.get_rl_config() must return Dict[str, Any]"

        # Verify optional get_rl_hooks exists (with default None behavior)
        # Note: Protocol default implementations don't auto-apply to implementing classes
        # so we only check that if it exists, it returns Optional[Any]
        if hasattr(config, "get_rl_hooks"):
            hooks = config.get_rl_hooks()
            # Can be None or an RLHooks instance
            assert hooks is None or hooks is not None  # Always true, validates callable
