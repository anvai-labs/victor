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

"""Dashboard views package.

This package contains specialized views for the observability dashboard:
- EventsView: Real-time and historical event viewing
- ToolsView: Tool execution statistics and history
- VerticalView: Vertical integration traces
- MetricsView: Performance metrics and charts
"""

from victor.observability.dashboard.views.events import (
    EventLogWidget,
    EventTableWidget,
    EventFilterWidget,
)
from victor.observability.dashboard.views.tools import (
    ToolStatsWidget,
    ToolHistoryWidget,
)
from victor.observability.dashboard.views.verticals import (
    VerticalTraceWidget,
    IntegrationResultWidget,
)

__all__ = [
    "EventLogWidget",
    "EventTableWidget",
    "EventFilterWidget",
    "ToolStatsWidget",
    "ToolHistoryWidget",
    "VerticalTraceWidget",
    "IntegrationResultWidget",
]
