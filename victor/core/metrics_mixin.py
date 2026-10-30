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

"""Shared ad-hoc ``record_metric`` surface for FEP-0014 metrics collectors.

Several concrete ``MetricsCollector`` classes satisfy
:class:`victor.core.protocols.MetricsCollectorProtocol` by storing generic
name/value observations next to their native, typed metrics. This mixin owns
that store and the ``record_metric`` adapter; each collector keeps its own
``get_snapshot`` shape.
"""

from __future__ import annotations

from typing import Dict


class AdhocMetricsMixin:
    """Provides ``record_metric`` backed by a lazily created ``_adhoc_metrics`` dict.

    The store is created on first access, so it exists regardless of where (or
    whether) a subclass ``__init__`` runs relative to other setup code.
    """

    _adhoc_metrics_store: Dict[str, float]

    @property
    def _adhoc_metrics(self) -> Dict[str, float]:
        """Ad-hoc metric values keyed by name (tag-qualified when tagged)."""
        try:
            return self._adhoc_metrics_store
        except AttributeError:
            store: Dict[str, float] = {}
            self._adhoc_metrics_store = store
            return store

    def record_metric(self, name: str, value: float, **tags: str) -> None:
        """Record a single ad-hoc metric observation (protocol adapter).

        Stores the value under ``name``, tag-qualified as
        ``name{k1=v1,k2=v2}`` (tags sorted by key) when tags are supplied.
        A later observation for the same key overwrites the earlier one.

        Args:
            name: Metric name.
            value: Numeric value to record (coerced to ``float``).
            **tags: Optional string-valued dimensional tags, appended to the key.
        """
        key = name
        if tags:
            tag_suffix = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            key = f"{name}{{{tag_suffix}}}"
        self._adhoc_metrics[key] = float(value)
