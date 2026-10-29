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

"""Sandbox settings in the subprocess config group."""

import pytest
from pydantic import ValidationError

from victor.config.groups.subprocess_config import SubprocessSettings


def test_sandbox_defaults():
    config = SubprocessSettings()
    assert config.code_executor_image == "python:3.11-slim"
    assert config.code_executor_idle_ttl_seconds == 900
    assert config.code_executor_network_disabled is True
    assert config.code_executor_memory_limit == "512m"
    assert config.code_executor_cpu_shares == 256


def test_idle_ttl_has_a_floor():
    # a TTL below the watchdog's polling cadence would expire mid-use
    with pytest.raises(ValidationError):
        SubprocessSettings(code_executor_idle_ttl_seconds=5)
    assert (
        SubprocessSettings(code_executor_idle_ttl_seconds=10).code_executor_idle_ttl_seconds == 10
    )


def test_settings_expose_the_group():
    from victor.config.settings import Settings

    settings = Settings()
    assert settings.subprocess.code_executor_idle_ttl_seconds == 900
    assert settings.subprocess.code_executor_image == "python:3.11-slim"
