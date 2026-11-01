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

Ollama and LM Studio discover their server the same way: an env-var CSV
overrides everything, then an explicit list/CSV/string base URL, then a
localhost default. The pair was duplicated verbatim in both providers.
"""

import os
from typing import List, Tuple, Union

BaseUrlLike = Union[None, str, List[str]]


def resolve_endpoint_candidates(
    base_url: BaseUrlLike,
    env_var: str,
    default_url: str,
) -> Tuple[List[str], Union[str, List[str], None]]:
    """Resolve the ordered list of endpoint candidates for a local provider.

    Priority: ``env_var`` CSV > explicit list/CSV/string ``base_url`` >
    ``default_url``.  Returns ``(candidates, base_url)`` with ``base_url``
    defaulted to ``default_url`` when passed as ``None`` (matching the
    historical inline behavior callers rely on).
    """
    candidates: List[str] = []

    # Check environment variable first
    env_endpoints = os.environ.get(env_var, "")
    if env_endpoints:
        candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

    if base_url is None:
        base_url = default_url

    # Only process base_url if no env var candidates
    if not candidates:
        if isinstance(base_url, (list, tuple)):
            candidates = [str(u).strip() for u in base_url if str(u).strip()]
        elif isinstance(base_url, str):
            if "," in base_url:
                candidates = [u.strip() for u in base_url.split(",") if u.strip()]
            else:
                candidates = [base_url]
        else:
            candidates = [str(base_url)]

    # An explicit-but-empty list must still yield the default (callers index
    # candidates[0] blindly; the pre-refactor inline code filled the default).
    if not candidates:
        candidates = [default_url]

    return candidates, base_url
