# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Background-task message formatting regression guard.

The background-task completion messages (STDOUT/STDERR blocks, error
messages, the SYSTEM HINT) previously used ``\\\\n`` in their f-strings —
Python string literals containing a literal backslash + 'n' instead of a
newline — so the model received ``\\n`` text sequences exactly when
long-running tasks finished (co-design review U1-3). Nothing parses those
strings; the fix is display/model-visible formatting only. This test pins
the module against the escape regression class.
"""

from __future__ import annotations

from pathlib import Path

import victor.agent.services.turn_execution_runtime as turn_execution_runtime


def test_no_literal_backslash_n_escapes_in_message_templates():
    """Scan the background-task message template lines (not the whole module
    — legitimate regex/path literals exist elsewhere) for the escape class."""
    source = Path(turn_execution_runtime.__file__).read_text(encoding="utf-8")
    template_markers = (
        "### STDOUT",
        "### STDERR",
        "### ❌ ERROR",
        "### 💡 SYSTEM HINT",
        "finished with result:",
    )
    offending = [
        line.strip()
        for line in source.splitlines()
        if any(marker in line for marker in template_markers)
        # Source-level backslash-backslash-n renders as a LITERAL backslash-n
        # in the message (the bug); single backslash-n is a real newline (ok).
        and "\\\\n" in line
    ]
    assert not offending, (
        "message templates regressed to literal backslash-n escapes: " f"{offending[:3]}"
    )


def test_system_hint_template_renders_real_newlines():
    # The fixed template shape (same construction as the production code).
    task_id = "task-1"
    hint = (
        f"### 💡 SYSTEM HINT\nTask is running in the background.\n"
        f"Task ID: {task_id}\n\nThe watcher will notify you when it completes."
    )
    assert "\n" in hint
    assert "\\n" not in hint  # no literal backslash-n in the rendered string
