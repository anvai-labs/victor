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

"""Canonical provider-classification constants (single source of truth).

Historically ~10 modules each declared their own inline "local providers"
set with divergent membership — llamacpp missing from the keyless set,
inferflux requiring an on-box GPU — so behavior drifted per site. Import
from here instead of restating sets:

- :data:`LOCAL_CLASS_PROVIDERS` — key-optional / local-class wire behavior:
  requests succeed without an ``Authorization`` header and are served by a
  self-hosted endpoint (on-box or a private network host such as the
  InferFlux SSH tunnel).
- :data:`ON_BOX_PROVIDERS` — inference runs on THIS machine's GPU: local
  resource detection (GPU presence) and single-tenant parallelism caps apply.
  InferFlux is deliberately NOT a member: it is a remote server and its
  availability says nothing about local GPUs.

Semantically distinct sets live where their semantics live (e.g.
``LOCAL_ENDPOINT_PROVIDERS`` in ``victor.framework.session_config`` for the
``--endpoint`` surface, which also encodes the 4s edge-decision budget) and
are not folded in here.
"""

from __future__ import annotations

# Key-optional / local-class: no API key required, self-hosted endpoint.
# (llamacpp was historically missing from copies of this set — a latent bug
# that left the stock llama-server path flagged keyless-unhealthy.)
LOCAL_CLASS_PROVIDERS = frozenset(
    {
        "ollama",
        "lmstudio",
        "vllm",
        "llamacpp",
        "inferflux",
    }
)

# Inference runs on this machine's GPU: single-tenant, locally detected.
ON_BOX_PROVIDERS = LOCAL_CLASS_PROVIDERS - {"inferflux"}


def is_local_class_provider(name: str) -> bool:
    """True when ``name`` is a key-optional, self-hosted-endpoint provider."""
    return bool(name) and name.lower() in LOCAL_CLASS_PROVIDERS


def is_on_box_provider(name: str) -> bool:
    """True when ``name``'s inference runs on this machine's GPU."""
    return bool(name) and name.lower() in ON_BOX_PROVIDERS


__all__ = [
    "LOCAL_CLASS_PROVIDERS",
    "ON_BOX_PROVIDERS",
    "is_local_class_provider",
    "is_on_box_provider",
]
