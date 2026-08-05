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

"""Victor Commands Module.

Provides shared command definitions and registry for consistent
behavior across all clients (CLI, VS Code, MCP).
"""

from victor.commands.shared_commands import (
    CommandCategory,
    CommandDefinition,
    CommandParameter,
    CommandRegistry,
    ParameterType,
    SHARED_COMMANDS,
    command_handler,
    command_registry,
)

__all__ = [
    "CommandCategory",
    "CommandDefinition",
    "CommandParameter",
    "CommandRegistry",
    "ParameterType",
    "SHARED_COMMANDS",
    "command_handler",
    "command_registry",
]
