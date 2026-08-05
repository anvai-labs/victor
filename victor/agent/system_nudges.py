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

"""Disposition of mid-conversation system nudges under prompt caching."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["log_dropped_system_nudge"]


def log_dropped_system_nudge(content: str) -> bool:
    """Record that a bracket-delimited system nudge was discarded. Returns True.

    Dropping it is correct: appending a second system message mid-conversation
    would break the byte-stable prefix that provider prompt caching depends on.
    But this used to happen via a bare ``return``, so a caller whose nudge
    happened to be bracket-delimited lost the message with no trace anywhere —
    no log line, no metric, nothing to correlate against a missing reminder.
    """
    logger.warning(
        "[cache] Dropping bracket-delimited system nudge to keep the cached "
        "prefix stable; it will not reach the model. Deliver it through the "
        "reminder manager instead. Content: %.120s",
        content,
    )
    return True
