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

"""Helpers for keeping log records bounded in size.

Error-path logs interpolate provider response bodies and exception text, which
can be arbitrarily large (a full HTTP error body, a multi-kilobyte traceback).
Emitted at ERROR/WARNING they survive log-level filtering, so a stuck loop that
spins on an error path can write those bodies to a redirected log unbounded.
``truncate_for_log`` caps that content to a sane ceiling while preserving how
much was elided (see TD-20 in docs/tech-stack.md).
"""

from __future__ import annotations

# Ceiling for content interpolated into a single log record. Chosen to keep an
# error body readable for diagnosis while bounding a spinning error path.
MAX_LOG_CHARS = 500


def truncate_for_log(text: object, limit: int = MAX_LOG_CHARS) -> str:
    """Bound arbitrary content destined for a log record.

    Args:
        text: Value to log; coerced to ``str`` (accepts exceptions directly).
        limit: Maximum characters to keep. Non-positive disables truncation.

    Returns:
        ``str(text)`` unchanged when within ``limit``; otherwise the first
        ``limit`` characters followed by a ``… (+N more chars)`` marker so the
        elision is visible in the log.
    """
    s = str(text)
    if limit <= 0 or len(s) <= limit:
        return s
    return f"{s[:limit]}… (+{len(s) - limit} more chars)"
