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

"""Default stage definitions for external verticals.

Returns SDK stage-definition contracts derived from the framework runtime
defaults so external verticals can import via::

    from victor.framework.defaults import get_default_stages
"""

from __future__ import annotations

from typing import Dict

from victor.framework.stage_manager import (
    get_default_stages as _get_runtime_default_stages,
    to_sdk_stage_definition,
)
from victor_contracts import StageDefinition


def get_default_stages() -> Dict[str, StageDefinition]:
    """Return SDK stage definitions derived from framework runtime defaults."""

    return {
        stage_name: to_sdk_stage_definition(stage_name, stage_definition)
        for stage_name, stage_definition in _get_runtime_default_stages().items()
    }


__all__ = [
    "get_default_stages",
]
