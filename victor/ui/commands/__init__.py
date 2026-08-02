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

"""Victor CLI commands package.

This package contains CLI commands (chat, init, tools, etc.) and
re-exports the modular slash command system.

The slash command system is implemented in victor.ui.slash/ using
SOLID principles for clean, extensible command handling.
"""

# Re-export from the modular slash command system
from victor.ui.slash import (
    SlashCommandHandler,
    SlashCommandProtocol,
    BaseSlashCommand,
    CommandContext,
    CommandMetadata,
    CommandRegistry,
    get_command_registry,
    register_command,
    command,
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
