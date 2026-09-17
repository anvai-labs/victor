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

"""Pins for the canonical provider-classification constants.

Before this module existed, ~10 modules each declared their own inline
"local providers" set with divergent membership (llamacpp missing from the
keyless set, inferflux requiring an on-box GPU). These tests pin the
canonical membership and the historical aliases that must stay in sync.
"""

from victor.config import api_keys
from victor.providers.provider_kinds import (
    LOCAL_CLASS_PROVIDERS,
    ON_BOX_PROVIDERS,
    is_local_class_provider,
    is_on_box_provider,
)


def test_local_class_membership():
    for name in ("ollama", "lmstudio", "vllm", "llamacpp", "inferflux"):
        assert is_local_class_provider(name), name


def test_local_class_excludes_cloud():
    for name in ("anthropic", "openai", "google", "", None):
        assert not is_local_class_provider(name)


def test_on_box_excludes_remote_inferflux():
    """InferFlux is a remote server: its availability says nothing about
    local GPUs, so resource detection must not gate on it."""
    assert "inferflux" not in ON_BOX_PROVIDERS
    for name in ("ollama", "lmstudio", "vllm", "llamacpp"):
        assert is_on_box_provider(name), name
    assert not is_on_box_provider("inferflux")
    assert not is_on_box_provider("anthropic")


def test_case_insensitive():
    assert is_local_class_provider("Ollama")
    assert is_on_box_provider("LLAMACPP")


def test_api_keys_alias_stays_in_sync():
    """api_keys.LOCAL_PROVIDERS is the historical import path (resolution,
    auth CLI, prompt builder) — it must alias the canonical set."""
    assert api_keys.LOCAL_PROVIDERS == set(LOCAL_CLASS_PROVIDERS)


def test_keyless_bug_fix_llamacpp_is_key_optional():
    """llamacpp was historically missing from the keyless set: the stock
    llama-server path was flagged keyless-unhealthy and skipped."""
    assert "llamacpp" in api_keys.LOCAL_PROVIDERS
