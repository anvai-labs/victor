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

"""Mixins for team coordinator composition.

These mixins add optional capabilities to team coordinators via composition,
following the ISP (Interface Segregation Principle) and favoring composition
over inheritance.

Mixins:
    ObservabilityMixin: Adds EventBus integration and progress tracking
    RLMixin: Adds RL integration for team composition learning
"""

from victor.teams.mixins.observability import ObservabilityMixin
from victor.teams.mixins.rl import RLMixin

__all__ = [
    "ObservabilityMixin",
    "RLMixin",
]
