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

"""Evaluation settings group (ADR-036 field-data collection).

The field-data recorder is opt-in: ``evaluation.field_data_enabled`` defaults
OFF so no 0.32 session is recorded unless the operator flips the flag.
"""

from __future__ import annotations

import pytest

from victor.config.groups.evaluation_config import EvaluationSettings


def test_field_data_defaults_off() -> None:
    group = EvaluationSettings()
    assert group.field_data_enabled is False
    assert group.field_data_store_dir is None
    assert group.field_data_max_file_bytes > 0


def test_field_data_group_is_a_nested_settings_group() -> None:
    from victor.config.settings import _NESTED_GROUPS

    assert _NESTED_GROUPS.get("evaluation") is EvaluationSettings


@pytest.mark.parametrize("env", ("1", "true", "yes", "on"))
def test_flat_field_env_override_enables_collection(
    monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    monkeypatch.setenv("VICTOR_SKIP_ENV_FILE", "1")
    monkeypatch.setenv("VICTOR_FIELD_DATA_ENABLED", env)
    from victor.config.settings import Settings

    settings = Settings()
    assert settings.evaluation.field_data_enabled is True


def test_flat_field_default_stays_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_SKIP_ENV_FILE", "1")
    monkeypatch.delenv("VICTOR_FIELD_DATA_ENABLED", raising=False)
    from victor.config.settings import Settings

    settings = Settings()
    assert settings.field_data_enabled is False
    assert settings.evaluation.field_data_enabled is False
