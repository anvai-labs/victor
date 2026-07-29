# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""code_generation_harness threads the profile's credential-identity account — TDD."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.evaluation.code_generation_harness import create_code_gen_runner


def test_create_code_gen_runner_threads_profile_account():
    """The profile's ``account`` must thread into provider settings so the eval
    harness resolves the profile's OWN key/endpoint, not the provider default."""
    profile_config = SimpleNamespace(
        provider="anthropic",
        model="glm-5.2",
        account="kimi-k3-anthropic",
    )
    mock_settings = MagicMock()
    mock_settings.load_profiles.return_value = {"kimi-anthropic": profile_config}
    mock_settings.get_provider_settings.return_value = {"api_key": "sk-moonshot"}

    with (
        patch("victor.config.settings.load_settings", return_value=mock_settings),
        patch("victor.providers.registry.ProviderRegistry.create", return_value=MagicMock()),
    ):
        create_code_gen_runner("kimi-anthropic")

    called = mock_settings.get_provider_settings.call_args
    assert called.args[0] == "anthropic"
    assert called.kwargs.get("account_name") == "kimi-k3-anthropic"
