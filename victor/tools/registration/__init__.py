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

"""Tool registration strategy pattern.

This package provides an extensible tool registration system that follows
the Open/Closed Principle (OCP). New tool types can be added without
modifying existing code by registering new strategies.

Architecture:
    ToolRegistrationStrategy (Protocol)
        ├── FunctionDecoratorStrategy
        ├── BaseToolSubclassStrategy
        └── MCPDictStrategy

    ToolRegistrationStrategyRegistry
        └── Manages strategy selection and registration

Usage:
    # Add a custom tool type without modifying core code
    class PydanticModelStrategy:
        def can_handle(self, tool):
            return isinstance(tool, BaseModel)

        def register(self, registry, tool, enabled=True):
            wrapper = self._create_wrapper(tool)
            registry._register_direct(wrapper.name, wrapper, enabled)

    # Register the strategy
    registry = ToolRegistry()
    registry.add_custom_strategy(PydanticModelStrategy())

    # Now you can register Pydantic models as tools
    registry.register(MyPydanticModel)
"""

from victor.tools.registration.strategies import (
    ToolRegistrationStrategy,
    FunctionDecoratorStrategy,
    BaseToolSubclassStrategy,
    MCPDictStrategy,
)
from victor.tools.registration.registry import ToolRegistrationStrategyRegistry

__all__ = [
    "ToolRegistrationStrategy",
    "FunctionDecoratorStrategy",
    "BaseToolSubclassStrategy",
    "MCPDictStrategy",
    "ToolRegistrationStrategyRegistry",
]
