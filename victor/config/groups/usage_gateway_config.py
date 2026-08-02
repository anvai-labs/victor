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

"""Usage-gateway (sandhi) attribution configuration settings.

FEP-0020 Phase 2 — controls the optional SandhiMeter attach at cost-tracker
build time. Default-off: when ``enabled`` is False the base install is
byte-identical (no sandhi import, no gateway construction).
"""

from typing import Optional

from pydantic import BaseModel


class UsageGatewaySettings(BaseModel):
    """FEP-0020 usage-gateway middleware attach (default-off: byte-identical when disabled)."""

    enabled: bool = False
    # None -> <global logs dir>/usage_events.jsonl derived at attach time.
    sink_path: Optional[str] = None
    # Operator-level attribution; the API-server auth seam wins later (only-when-None fill).
    subject_id: Optional[str] = None
    group_id: Optional[str] = None
