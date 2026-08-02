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

"""Core safety types used across safety modules.

This module defines the fundamental types used by safety modules,
placed here to avoid circular imports between safety and verticals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyPattern:
    """A safety pattern for detecting dangerous operations.

    Attributes:
        pattern: Regex pattern to match
        description: Human-readable description
        risk_level: Risk level (use string for flexibility)
        category: Category of the pattern (e.g., "git", "filesystem")
    """

    pattern: str
    description: str
    risk_level: str = "HIGH"  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    category: str = "general"


__all__ = [
    "SafetyPattern",
]
