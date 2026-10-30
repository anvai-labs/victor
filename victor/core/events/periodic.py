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

"""Daemon-thread base for periodic metrics-snapshot reporters.

Shared by the sync-emitter and extension-loader reporters: an idempotent
``start``, a joinable ``stop`` and a loop that calls :meth:`_emit` every
``interval_seconds`` and never lets an emission error kill the thread.
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar, Optional


class PeriodicMetricsReporter:
    """Periodically publish a metrics snapshot from a daemon thread.

    Subclasses set ``_THREAD_NAME`` and implement :meth:`_emit`.
    """

    _THREAD_NAME: ClassVar[str] = "periodic-metrics-reporter"

    def __init__(
        self,
        *,
        interval_seconds: float,
        topic: str,
        source: str,
        reset_after_emit: bool,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")

        self._interval_seconds = interval_seconds
        self._topic = topic
        self._source = source
        self._reset_after_emit = reset_after_emit
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger(type(self).__module__)

    @property
    def is_running(self) -> bool:
        """Return True while the reporter thread is active."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start periodic emission if not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=self._THREAD_NAME,
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop periodic emission and join the thread."""
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout=timeout)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                self._emit()
            except Exception as e:
                self._logger.debug("%s loop error: %s", type(self).__name__, e)

    def _emit(self) -> None:
        """Publish one metrics snapshot."""
        raise NotImplementedError
