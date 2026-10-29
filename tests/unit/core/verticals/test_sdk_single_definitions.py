# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Types promoted to the SDK are defined there once; core and framework re-export them.

Each of these used to be a separate copy in core/framework (identical apart
from cosmetics), so an object built from one import path failed isinstance
checks against the other and the copies could drift.
"""

import importlib

import pytest

PAIRS = [
    ("victor.core.mode_config", "ModeLevel", "victor_contracts.verticals.mode_config", "ModeLevel"),
    (
        "victor.core.mode_config",
        "ModeConfig",
        "victor_contracts.verticals.mode_config",
        "ModeConfig",
    ),
    (
        "victor.core.mode_config",
        "ModeDefinition",
        "victor_contracts.verticals.mode_config",
        "ModeDefinition",
    ),
    (
        "victor.core.mode_config",
        "VerticalModeConfig",
        "victor_contracts.verticals.mode_config",
        "VerticalModeConfig",
    ),
    (
        "victor.framework.vertical_protocols",
        "CCGBuilderProtocol",
        "victor_contracts.verticals.protocols.storage",
        "CCGBuilderProtocol",
    ),
]


@pytest.mark.parametrize("core_mod, core_name, sdk_mod, sdk_name", PAIRS)
def test_core_name_is_the_sdk_object(core_mod, core_name, sdk_mod, sdk_name):
    core = getattr(importlib.import_module(core_mod), core_name)
    sdk = getattr(importlib.import_module(sdk_mod), sdk_name)
    assert core is sdk


def test_objects_interoperate_across_import_paths():
    from victor_contracts.verticals.mode_config import ModeConfig as SdkModeConfig

    from victor.core.mode_config import DEFAULT_MODES, ModeConfig

    config = ModeConfig(tool_budget=5, max_iterations=3)
    assert isinstance(config, SdkModeConfig)
    assert all(isinstance(mode.to_mode_config(), SdkModeConfig) for mode in DEFAULT_MODES.values())


def test_ccg_registry_key_is_shared():
    """Registration and lookup both use the framework import path; it must be the SDK key."""
    from victor_contracts.verticals.protocols.storage import CCGBuilderProtocol as Sdk

    from victor.framework.vertical_protocols import CCGBuilderProtocol

    assert {CCGBuilderProtocol: "x"}[Sdk] == "x"
