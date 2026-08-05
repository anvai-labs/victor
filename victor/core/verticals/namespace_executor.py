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

"""Namespace-scoped executor pools for vertical isolation.

This keeps background sync work from every vertical on the global default
executor. It does not provide process isolation, but it gives each plugin
namespace its own bounded thread pool and therefore a smaller blast radius.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import Dict, Optional


class NamespaceExecutorPool:
    """Manage one ThreadPoolExecutor per plugin namespace."""

    _instance: Optional["NamespaceExecutorPool"] = None
    _instance_lock = threading.RLock()

    def __init__(self, max_workers_per_namespace: Optional[int] = None) -> None:
        cpu_count = os.cpu_count() or 1
        self._max_workers = max_workers_per_namespace or max(1, min(4, cpu_count))
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "NamespaceExecutorPool":
        """Return the singleton pool."""

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_executor(self, namespace: str) -> ThreadPoolExecutor:
        """Return the executor for *namespace*, creating it on first use."""

        normalized = namespace or "default"
        with self._lock:
            executor = self._executors.get(normalized)
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix=f"victor-{normalized}",
                )
                self._executors[normalized] = executor
            return executor

    def clear(self, *, shutdown: bool = True) -> int:
        """Clear all namespace executors, optionally shutting them down."""

        with self._lock:
            executors = list(self._executors.values())
            count = len(executors)
            self._executors.clear()

        if shutdown:
            for executor in executors:
                executor.shutdown(wait=False)
        return count

    def executor_count(self) -> int:
        """Return the number of namespace executors currently allocated."""

        with self._lock:
            return len(self._executors)


def get_namespace_executor_pool() -> NamespaceExecutorPool:
    """Return the shared namespace executor pool."""

    return NamespaceExecutorPool.get_instance()


__all__ = ["NamespaceExecutorPool", "get_namespace_executor_pool"]
