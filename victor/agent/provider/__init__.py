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

"""Provider management components.

This package provides focused components for provider management,
extracted from ProviderManager to follow the Single Responsibility
Principle (SRP).

Migration Note (2026-05-04):
- ProviderCoordinator and ProviderSwitchCoordinator removed in v1.0.0
- Use ProviderService for all provider operations
"""

from victor.agent.provider.health_monitor import ProviderHealthMonitor
from victor.agent.provider.switcher import (
    ProviderSwitcher,
    ProviderSwitcherState,
)
from victor.agent.provider.tool_adapter_coordinator import ToolAdapterCoordinator

__all__ = [
    "ProviderHealthMonitor",
    "ProviderSwitcher",
    "ProviderSwitcherState",
    "ToolAdapterCoordinator",
]
