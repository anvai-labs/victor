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

"""Legacy turn executor compatibility import path.

The canonical implementation lives in
``victor.agent.services.turn_execution_runtime``. This module remains as a
thin re-export so older imports keep working during the service-first
migration.
"""

from victor.agent.services.turn_execution_runtime import TurnExecutor, TurnResult

__all__ = ["TurnExecutor", "TurnResult"]
