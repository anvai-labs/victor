# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Process-cached settings snapshot (load_settings).

Settings construction costs ~5 ms (.env parse + double os.environ scan +
155-field pydantic validation) and used to run per LLM request and per
graph invocation (co-design review U3-F1/U2-F3). load_settings() now
returns a shared read-only snapshot; mutators construct Settings()
directly; reset_settings_cache() is the invalidation hook.
"""

from __future__ import annotations

import pytest

import victor.config.settings as settings_module
from victor.config.settings import (
    Settings,
    get_settings,
    load_settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


class TestSettingsSnapshotCache:
    def test_load_settings_returns_same_instance(self):
        assert load_settings() is load_settings()

    def test_get_settings_alias_shares_cache(self):
        assert get_settings() is load_settings()

    def test_direct_construction_is_uncached(self):
        assert Settings() is not load_settings()
        assert Settings() is not Settings()

    def test_reset_rebuilds_snapshot(self):
        first = load_settings()
        reset_settings_cache()
        assert load_settings() is not first

    def test_fresh_bypasses_cache_without_disturbing_it(self):
        cached = load_settings()
        fresh = load_settings(fresh=True)
        assert fresh is not cached
        assert load_settings() is cached  # snapshot untouched by fresh builds

    def test_env_change_invisible_until_reset(self, monkeypatch):
        # Env writes after the snapshot exists must NOT leak in...
        cached = load_settings()
        monkeypatch.setenv("VICTOR_LOG_LEVEL", "DEBUG")
        assert load_settings() is cached
        # ...and become visible after reset.
        reset_settings_cache()
        assert load_settings() is not cached

    def test_symbol_remains_patchable(self, monkeypatch):
        """test_sandhi_transport patches the get_settings symbol — the module
        attribute indirection must keep working."""
        sentinel = object()
        monkeypatch.setattr(settings_module, "get_settings", lambda: sentinel)
        assert settings_module.get_settings() is sentinel

    def test_snapshot_read_is_stable_under_repeated_calls(self):
        a, b = load_settings(), get_settings()
        assert type(a) is Settings
        assert a is b
