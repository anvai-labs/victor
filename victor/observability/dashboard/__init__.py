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

"""Victor Observability Dashboard - TUI for visualizing events and traces.

This module provides a Textual-based TUI dashboard for:
- Real-time event streaming visualization
- Integration result browsing from JSONL logs
- Vertical configuration trace display
- Tool execution history

Example:
    from victor.observability.dashboard import ObservabilityDashboard

    # Run the dashboard
    app = ObservabilityDashboard()
    await app.run_async()

    # Or with a specific log file
    app = ObservabilityDashboard(log_file="events.jsonl")
    await app.run_async()
"""

from victor.observability.dashboard.app import ObservabilityDashboard, run_dashboard

__all__ = ["ObservabilityDashboard", "run_dashboard"]
