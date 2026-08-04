# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""``PromptInput`` — the chat prompt with OS-clipboard-aware, multi-line paste.

Textual's stock ``Input`` mishandles paste for a chat prompt in two ways:

* **Ctrl+V** (``action_paste``) reads Textual's in-app clipboard buffer, not the
  operating-system clipboard, so it does nothing for text copied in another app.
* **Bracketed paste** (terminal Ctrl+Shift+V / right-click) keeps only the first
  line — multi-line messages are silently truncated.

``PromptInput`` fixes both: Ctrl+V reads the real OS clipboard (see
:mod:`victor.ui.tui.clipboard`), and bracketed paste inserts the full payload.
The value is single-line to render but preserves embedded newlines, so a pasted
multi-line message submits verbatim.
"""

from __future__ import annotations

import asyncio

from textual import events
from textual.widgets import Input

from victor.ui.tui.clipboard import read_clipboard


class PromptInput(Input):
    """An ``Input`` whose paste reads the OS clipboard and keeps every line."""

    def _replace_selection(self, text: str) -> None:
        """Insert ``text`` at the cursor, replacing any active selection."""
        if not text:
            return
        start, end = self.selection
        self.replace(text, start, end)

    async def action_paste(self) -> None:
        """Paste the OS clipboard on Ctrl+V, falling back to the in-app buffer.

        The clipboard read shells out to a platform tool, so it runs in a worker
        thread to avoid blocking the event loop.
        """
        text = await asyncio.to_thread(read_clipboard)
        if not text:
            text = self.app.clipboard  # text copied earlier inside the app
        self._replace_selection(text)

    def _on_paste(self, event: events.Paste) -> None:
        """Insert the full bracketed-paste payload (Textual keeps only line one)."""
        if event.text:
            self._replace_selection(event.text)
        event.stop()
