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

"""Service-owned tool runtime contract value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ToolCallValidation:
    """Result of validating a single tool call."""

    valid: bool
    original_name: Optional[str] = None
    canonical_name: Optional[str] = None
    skip_reason: Optional[str] = None
    error_result: Optional[Dict[str, Any]] = None


@dataclass
class NormalizedArgs:
    """Result of full tool argument normalization."""

    args: Dict[str, Any]
    strategy: Any
    signature: Tuple[str, str]
    is_repeated_failure: bool = False


__all__ = ["ToolCallValidation", "NormalizedArgs"]
