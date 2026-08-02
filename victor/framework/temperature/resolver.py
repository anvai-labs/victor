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

"""The composition root for unified temperature resolution (ADR-013).

``resolve`` picks a base from the first source that returns non-``None`` (Chain of Responsibility by
precedence), guarantees a value via a terminal global-default fallback, then applies modifiers in
order — recording a ``modifier_trace`` for observability. Extending the policy means appending a
source/modifier in :func:`build_default_resolver`; this class never changes (OCP), and it depends only
on the two Protocols (DIP).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from victor.framework.temperature.defaults import GLOBAL_DEFAULT
from victor.framework.temperature.protocols import (
    TemperatureContext,
    TemperatureModifier,
    TemperatureRequest,
    TemperatureResolution,
    TemperatureSource,
)

logger = logging.getLogger(__name__)


class TemperatureResolver:
    """Resolve an effective sampling temperature from composed sources + modifiers."""

    def __init__(
        self,
        sources: List[TemperatureSource],
        modifiers: List[TemperatureModifier],
        *,
        global_default: float = GLOBAL_DEFAULT,
    ) -> None:
        self._sources = list(sources)
        self._modifiers = list(modifiers)
        self._global_default = global_default

    def resolve(
        self, request: TemperatureRequest, context: Optional[TemperatureContext] = None
    ) -> TemperatureResolution:
        ctx = context or TemperatureContext()

        base: Optional[float] = None
        source_name = "global_default_fallback"
        for source in self._sources:
            value = source.resolve(request)
            if value is not None:
                base, source_name = float(value), source.name
                break
        if base is None:
            base = self._global_default

        value = base
        trace = []
        for modifier in self._modifiers:
            new_value, reason = modifier.adjust(value, request, ctx)
            trace.append((modifier.name, float(new_value), reason))
            value = float(new_value)

        resolution = TemperatureResolution(
            value=value, base=base, source_name=source_name, modifier_trace=tuple(trace)
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[Temperature] %s", resolution.to_dict())
        return resolution
