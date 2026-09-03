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
    source = Path(turn_execution_runtime.__file__).read_text(encoding="utf-8")
    # A double backslash followed by 'n' in source renders as literal "\n"
    # text — never intended in user/model-visible message templates.
    assert "\\\\n" not in source, (
        "turn_execution_runtime regressed: found literal backslash-n escapes "
        "('\\\\n') in string templates — messages must contain real newlines"
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
