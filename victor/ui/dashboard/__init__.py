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

"""Victor Metrics Dashboard

Provides real-time metrics visualization and historical data analysis
for agent operations.

Usage:
    victor observability dashboard
    victor observability metrics
"""

from victor.ui.dashboard.display import MetricsDashboard
from victor.ui.dashboard.data import DashboardDataProvider

__all__ = [
    "MetricsDashboard",
    "DashboardDataProvider",
]
