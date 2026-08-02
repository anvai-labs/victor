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

"""Slash command handling for Victor CLI.

This module re-exports the modular slash command system from victor.ui.slash.

The implementation uses a SOLID-based, protocol-driven architecture that
enables easy addition of new commands without modifying existing code.

For adding new commands, see victor.ui.slash.commands/ for examples.
"""

# Re-export everything from the modular system
from victor.ui.slash import (
    BaseSlashCommand,
    CommandContext,
    CommandMetadata,
    CommandRegistry,
    SlashCommandHandler,
    SlashCommandProtocol,
    command,
    get_command_registry,
    register_command,
)

__all__ = [
    "SlashCommandHandler",
    "SlashCommandProtocol",
    "BaseSlashCommand",
    "CommandContext",
    "CommandMetadata",
    "CommandRegistry",
    "get_command_registry",
    "register_command",
    "command",
]
