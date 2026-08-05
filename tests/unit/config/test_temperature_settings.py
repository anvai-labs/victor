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

"""Tests for TemperatureSettings + ProfileConfig.temperatures + the settings→resolver adapter (PR-B)."""

from victor.config.settings import ProfileConfig, Settings
from victor.config.temperature_settings import TemperatureSettings
from victor.framework.temperature import build_resolver_from_settings
from victor.framework.temperature.protocols import TemperatureRequest


def test_temperature_settings_defaults():
    ts = TemperatureSettings()
    assert ts.global_default == 0.6
    assert ts.ratchet_step == 0.05 and ts.ratchet_cap == 0.9
    assert ts.proactive_ratchet_enabled is True
    assert ts.task_defaults == {}


def test_settings_exposes_temperature_group():
    settings = Settings()
    # nested group auto-instantiates from defaults; named temperature_policy to avoid colliding
    # with the float temperature that callers read via getattr(settings, "temperature", 0.7).
    assert isinstance(settings.temperature_policy, TemperatureSettings)
    assert settings.temperature_policy.global_default == 0.6
    # the float `temperature` reader path must NOT receive the group object (the regression we fixed)
    assert not isinstance(getattr(settings, "temperature", None), TemperatureSettings)


def test_profile_config_accepts_temperatures_map():
    profile = ProfileConfig(provider="zai", model="glm-5.2", temperatures={"plan": 0.5})
    assert profile.temperatures == {"plan": 0.5}
    # absent → None (defer down the chain)
    assert ProfileConfig(provider="zai", model="glm-5.2").temperatures is None


def test_adapter_maps_settings_to_resolver():
    ts = TemperatureSettings(global_default=0.55, ratchet_step=0.1, ratchet_cap=0.8)
    resolver = build_resolver_from_settings(ts)
    # global default flows through (no source resolves → terminal)
    assert resolver.resolve(TemperatureRequest(model_name="claude")).value == 0.55


def test_adapter_disables_ratchet_when_settings_say_so():
    ts = TemperatureSettings(proactive_ratchet_enabled=False)
    resolver = build_resolver_from_settings(ts)
    # find the ratchet modifier and confirm it is disabled
    ratchet = next(m for m in resolver._modifiers if m.name == "spin_ratchet")
    assert ratchet.enabled is False
