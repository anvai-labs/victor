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

"""Vertical composition framework.

This package provides composition-based architecture for building
verticals, following the Open/Closed Principle (OCP) and
Interface Segregation Principle (ISP).

Instead of inheriting from VerticalBase and being forced to
implement or override many methods, verticals can now be built
by composing together only the capabilities they need.

Key Features:
- BaseComposer: Foundation for composing capabilities
- CapabilityComposer: Compose vertical capabilities declaratively
- Capability Providers: Focused, single-responsibility capability classes

Example:
    MyVertical = (
        VerticalBase
        .compose()
        .with_metadata("my_vertical", "My assistant", "1.0.0")
        .with_stages(custom_stages)
        .with_extensions([SafetyExtension(), LoggingExtension()])
        .build()
    )
"""

from victor.core.verticals.composition.base_composer import (
    BaseComposer,
    BaseCapabilityProvider,
)
from victor.core.verticals.composition.capability_composer import (
    CapabilityComposer,
    MetadataCapability,
    StagesCapability,
    ExtensionsCapability,
)

__all__ = [
    "BaseComposer",
    "BaseCapabilityProvider",
    "CapabilityComposer",
    "MetadataCapability",
    "StagesCapability",
    "ExtensionsCapability",
]
