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

"""Modular slash command system for Victor CLI.

This package provides a SOLID-based, protocol-driven slash command system
that enables easy addition of new commands without modifying existing code.

Architecture:
    - protocol.py: SlashCommandProtocol interface and CommandContext
    - registry.py: CommandRegistry for dynamic command registration
    - handler.py: SlashCommandHandler for command execution
    - commands/: Individual command implementations by category

Usage:
    from victor.ui.slash import SlashCommandHandler

    handler = SlashCommandHandler(console, settings, agent)
    await handler.execute("/help")

Adding new commands:
    1. Create a new class implementing SlashCommandProtocol
    2. Use @register_command decorator for auto-registration
    3. Place in appropriate category module under commands/

Example:
    from victor.ui.slash.protocol import CommandContext, CommandMetadata
    from victor.ui.slash.registry import register_command

    @register_command
    class MyCommand:
        @property
        def metadata(self) -> CommandMetadata:
            return CommandMetadata(
                name="mycommand",
                description="Does something useful",
                usage="/mycommand [args]",
                category="tools",
            )

        def execute(self, ctx: CommandContext) -> None:
            ctx.console.print("Hello from mycommand!")
"""

from victor.ui.slash.handler import SlashCommandHandler
from victor.ui.slash.protocol import (
    BaseSlashCommand,
    CommandContext,
    CommandMetadata,
    SlashCommandProtocol,
)
from victor.ui.slash.registry import (
    CommandRegistry,
    command,
    get_command_registry,
    register_command,
)

# NOTE: commands are intentionally NOT imported here (co-design review
# item 21). `from victor.ui.slash import commands` pulled the heaviest
# command module's import chain (bayesian -> framework.rl.monitoring ->
# victor.agent, ~0.9s measured) into every `victor.ui.slash` import,
# including `victor --help`. Registration now happens lazily: the first
# read on a CommandRegistry (get/has/list_commands/iter_commands/
# categories/list_by_category) triggers CommandRegistry._ensure_discovered
# (registry.py), and SlashCommandHandler.__init__ discovers eagerly at
# handler-construction time regardless. `victor.ui.slash.commands` and
# its submodules remain directly importable for tests/tooling that want
# them explicitly.

__all__ = [
    # Main handler
    "SlashCommandHandler",
    # Protocol and context
    "SlashCommandProtocol",
    "BaseSlashCommand",
    "CommandContext",
    "CommandMetadata",
    # Registry
    "CommandRegistry",
    "get_command_registry",
    "register_command",
    "command",
]
