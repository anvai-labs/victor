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

"""Observability event emitters for Victor.

This package provides modular, SOLID-compliant event emitters for different
categories of events. Each emitter is focused, testable, and extensible.

Emitters:
    - ToolEventEmitter: Tool execution tracking
    - ModelEventEmitter: LLM interaction tracking
    - StateEventEmitter: State transition tracking
    - LifecycleEventEmitter: Session lifecycle tracking
    - ErrorEventEmitter: Error tracking

All emitters implement Protocol-based interfaces for type safety and substitutability.
"""

from victor.observability.emitters.base import (
    IEventEmitter,
    IToolEventEmitter,
    IModelEventEmitter,
    IStateEventEmitter,
    ILifecycleEventEmitter,
    IErrorEventEmitter,
)
from victor.observability.emitters.tool_emitter import ToolEventEmitter
from victor.observability.emitters.model_emitter import ModelEventEmitter
from victor.observability.emitters.state_emitter import StateEventEmitter
from victor.observability.emitters.lifecycle_emitter import LifecycleEventEmitter
from victor.observability.emitters.error_emitter import ErrorEventEmitter

__all__ = [
    # Protocols
    "IEventEmitter",
    "IToolEventEmitter",
    "IModelEventEmitter",
    "IStateEventEmitter",
    "ILifecycleEventEmitter",
    "IErrorEventEmitter",
    # Implementations
    "ToolEventEmitter",
    "ModelEventEmitter",
    "StateEventEmitter",
    "LifecycleEventEmitter",
    "ErrorEventEmitter",
]
