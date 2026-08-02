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

"""Sandhi model-catalog consumption (TD-0004 Phase A).

Sandhi owns catalog **data** -- curated, versioned model facts (id, context
window, max output; no pricing) exposed via
``sandhi_gateway.provider_models_json``. Victor owns catalog **policy**: this
module shapes the neutral ``ModelDescriptorV1`` facts into Victor's model-dict
surface, and every consumer (Anthropic/OpenAI/Google provider shells) falls
back to its own SDK discovery / static list when the installed Sandhi binding
predates the catalog surface.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def models_from_sandhi(provider_slug: str) -> Optional[List[Dict[str, Any]]]:
    """Victor-shaped models from the Sandhi catalog, or ``None`` to fall back.

    Returns ``None`` (never raises) when sandhi-gateway is absent, predates the
    catalog surface, or has no data for ``provider_slug`` -- callers fall
    through to their live-SDK / static tiers.
    """
    try:
        import json

        import sandhi_gateway as sg  # lazy: only needed for discovery
    except Exception:
        return None
    if not hasattr(sg, "provider_models_json"):
        return None
    try:
        raw = json.loads(sg.provider_models_json(provider_slug))
    except Exception as exc:  # unknown provider, deserialize error, FFI failure
        logger.debug("sandhi catalog lookup failed for %s: %s", provider_slug, exc)
        return None
    models: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        extensions = entry.get("extensions") or {}
        models.append(
            {
                "id": entry["id"],
                "name": extensions.get("display_name") or entry["id"],
                "context_window": entry.get("max_input_tokens"),
                "max_output_tokens": entry.get("max_output_tokens"),
            }
        )
    return models or None
