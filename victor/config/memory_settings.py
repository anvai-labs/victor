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

"""Core long-term memory settings (FEP-0048)."""

from __future__ import annotations

from pydantic import BaseModel


class MemorySettings(BaseModel):
    """Core long-term memory behavior (FEP-0048).

    Write-path defaults are conservative: wrong memories actively harm every
    later session, so capture is human-confirmed unless explicitly opted in.
    """

    #: End-of-session consolidation auto-captures proposed episodes without
    #: confirmation. Default OFF — the Session-service hook proposes and
    #: stages episodes, but stores nothing until the confirm path runs
    #: (aggressive auto-capture is opt-in per FEP-0048's confirm-by-default
    #: posture). Env: VICTOR_MEMORY_AUTO_CAPTURE.
    memory_auto_capture: bool = False
