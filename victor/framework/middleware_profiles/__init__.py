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

"""Framework-level middleware profiles and builders.

This package provides pre-configured middleware profiles and builders
that can be used across all verticals to reduce duplication and ensure consistency.

Modules:
    profiles: Pre-defined middleware profiles for common use cases
    builder: Builder pattern for custom middleware profiles
"""

from __future__ import annotations

from victor.framework.middleware_profiles.profiles import (
    MiddlewareProfile,
    MiddlewareProfiles,
    get_profile,
    list_profiles,
)
from victor.framework.middleware_profiles.builder import (
    MiddlewareProfileBuilder,
    create_profile,
)

__all__ = [
    # Profiles
    "MiddlewareProfile",
    "MiddlewareProfiles",
    "get_profile",
    "list_profiles",
    # Builder
    "MiddlewareProfileBuilder",
    "create_profile",
]
