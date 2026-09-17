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

"""FIFO of user prompts awaiting execution (pure logic — no Textual imports).

The prompt stays live while a turn runs: submissions land here instead of
starting a second turn, and are consumed strictly in order when the current
run finishes (the app's turn-finished handler is the listener that drains
this queue). The queue is a temp structure by design — it lives for the app
session only and is dropped on exit.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional


class PromptQueue:
    """Ordered holding area for prompts submitted mid-turn (listener pattern).

    Args:
        on_change: Invoked after every state-changing mutation (successful
            enqueue, consume, non-empty clear). The app adapts this to a
            Textual message so UI projection stays on the message loop.
    """

    def __init__(self, on_change: Optional[Callable[["PromptQueue"], None]] = None) -> None:
        self._items: deque[str] = deque()
        self._on_change = on_change

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def enqueue(self, text: str) -> bool:
        """Add a prompt to the back of the queue.

        Whitespace-only input is ignored. Returns True when stored, and fires
        the ``on_change`` listener.
        """
        if not text or not text.strip():
            return False
        self._items.append(text)
        self._notify()
        return True

    def peek(self) -> Optional[str]:
        """Return the oldest queued prompt without removing it."""
        return self._items[0] if self._items else None

    def consume(self) -> Optional[str]:
        """Remove and return the oldest queued prompt (strict FIFO)."""
        if not self._items:
            return None
        text = self._items.popleft()
        self._notify()
        return text

    def clear(self) -> int:
        """Drop every queued prompt; returns how many were dropped."""
        dropped = len(self._items)
        if dropped:
            self._items.clear()
            self._notify()
        return dropped

    def snapshot(self) -> tuple[str, ...]:
        """Oldest-first view for display."""
        return tuple(self._items)

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change(self)
