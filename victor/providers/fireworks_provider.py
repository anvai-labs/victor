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

"""Fireworks AI policy over Victor's shared OpenAI-compatible adapter."""

from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("fireworks")
DEFAULT_BASE_URL = _SPEC.base_url
FIREWORKS_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}


class FireworksProvider(SandhiOpenAICompatPolicy):
    """Fireworks provider with declarative static policy and shared wire behavior."""

    CONFIG_KEY = "fireworks"


__all__ = ["DEFAULT_BASE_URL", "FIREWORKS_MODELS", "FireworksProvider"]
