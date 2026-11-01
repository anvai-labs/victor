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

"""Shared endpoint-candidate resolution for local providers.

The helper is the single implementation behind Ollama's and LM Studio's
tiered endpoint discovery (env var > explicit base URL > default), extracted
when their duplicated inline copies were consolidated.
"""

import pytest

from victor.providers.endpoint_resolution import resolve_endpoint_candidates


DEFAULT = "http://localhost:11434"


def _resolve(base_url, monkeypatch=None, env=None):
    if env is not None:
        monkeypatch.setenv("TEST_ENDPOINTS", env)
    else:
        monkeypatch.delenv("TEST_ENDPOINTS", raising=False)
    return resolve_endpoint_candidates(base_url, "TEST_ENDPOINTS", DEFAULT)


def test_env_var_csv_overrides_everything(monkeypatch):
    candidates, base_url = _resolve(
        "http://explicit:1234", monkeypatch, env=" http://a:1, http://b:2 ,"
    )
    assert candidates == ["http://a:1", "http://b:2"]
    # base_url is untouched when it was not None (callers may still use it).
    assert base_url == "http://explicit:1234"


def test_none_base_url_defaults(monkeypatch):
    candidates, base_url = _resolve(None, monkeypatch)
    assert base_url == DEFAULT
    assert candidates == [DEFAULT]


def test_list_base_url_preserved_and_split(monkeypatch):
    candidates, base_url = _resolve(["http://a:1", " http://b:2 ", ""], monkeypatch)
    assert candidates == ["http://a:1", "http://b:2"]
    assert base_url == ["http://a:1", " http://b:2 ", ""]


def test_csv_string_base_url_splits(monkeypatch):
    candidates, _ = _resolve("http://a:1,http://b:2", monkeypatch)
    assert candidates == ["http://a:1", "http://b:2"]


def test_plain_string_base_url(monkeypatch):
    candidates, _ = _resolve("http://only:1", monkeypatch)
    assert candidates == ["http://only:1"]


def test_empty_list_still_yields_default(monkeypatch):
    """Explicit-but-empty candidate lists must default: callers index
    candidates[0] blindly (the pre-extraction inline code filled the default)."""
    candidates, _ = _resolve([], monkeypatch)
    assert candidates == [DEFAULT]


def test_whitespace_only_csv_yields_default(monkeypatch):
    candidates, _ = _resolve(" , ", monkeypatch)
    assert candidates == [DEFAULT]


def test_unexpected_type_is_coerced(monkeypatch):
    candidates, _ = _resolve(1234, monkeypatch)  # type: ignore[arg-type]
    assert candidates == ["1234"]


@pytest.mark.parametrize("env_value", ["", "   "])
def test_blank_env_var_falls_through_to_base_url(monkeypatch, env_value):
    candidates, _ = _resolve("http://explicit:1", monkeypatch, env=env_value)
    assert candidates == ["http://explicit:1"]
