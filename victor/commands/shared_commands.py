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

"""Shared Command Definitions.

Provides a unified command interface that works across all clients:
- CLI (Python)
- VS Code Extension (TypeScript)
- MCP Clients

Commands are defined as JSON schemas with metadata for:
- Command name and description
- Parameters with validation
- Category for grouping
- Keyboard shortcuts (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CommandCategory(str, Enum):
    """Categories for grouping commands."""

    SESSION = "session"
    PROVIDER = "provider"
    TOOL = "tool"
    MODE = "mode"
    CONTEXT = "context"
    HELP = "help"
    DEBUG = "debug"


class ParameterType(str, Enum):
    """Types for command parameters."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    FILE = "file"
    DIRECTORY = "directory"


@dataclass
class CommandParameter:
    """Definition of a command parameter."""

    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Optional[Any] = None
    choices: Optional[List[str]] = None
    validation_pattern: Optional[str] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "description": self.description,
        }

        if self.type == ParameterType.STRING:
            schema["type"] = "string"
        elif self.type == ParameterType.NUMBER:
            schema["type"] = "number"
        elif self.type == ParameterType.BOOLEAN:
            schema["type"] = "boolean"
        elif self.type == ParameterType.CHOICE:
            schema["type"] = "string"
            if self.choices:
                schema["enum"] = self.choices
        elif self.type in (ParameterType.FILE, ParameterType.DIRECTORY):
            schema["type"] = "string"
            schema["format"] = "path"

        if self.default is not None:
            schema["default"] = self.default

        if self.validation_pattern:
            schema["pattern"] = self.validation_pattern

        return schema


@dataclass
class CommandDefinition:
    """Definition of a slash command."""

    name: str
    description: str
    category: CommandCategory
    parameters: List[CommandParameter] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    shortcut: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    hidden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "choices": p.choices,
                }
                for p in self.parameters
            ],
            "aliases": self.aliases,
            "shortcut": self.shortcut,
            "examples": self.examples,
            "hidden": self.hidden,
        }

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for MCP."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


# Built-in command definitions
SHARED_COMMANDS: List[CommandDefinition] = [
    # Session commands
    CommandDefinition(
        name="clear",
        description="Clear the conversation history",
        category=CommandCategory.SESSION,
        aliases=["reset"],
        shortcut="Ctrl+L",
        examples=["/clear"],
    ),
    CommandDefinition(
        name="save",
        description="Save the current session",
        category=CommandCategory.SESSION,
        parameters=[
            CommandParameter(
                name="name",
                type=ParameterType.STRING,
                description="Session name",
                required=False,
            ),
        ],
        examples=["/save", "/save my-session"],
    ),
    CommandDefinition(
        name="load",
        description="Load a saved session",
        category=CommandCategory.SESSION,
        parameters=[
            CommandParameter(
                name="name",
                type=ParameterType.STRING,
                description="Session name to load",
                required=True,
            ),
        ],
        examples=["/load my-session"],
    ),
    CommandDefinition(
        name="export",
        description="Export conversation to markdown or JSON",
        category=CommandCategory.SESSION,
        parameters=[
            CommandParameter(
                name="format",
                type=ParameterType.CHOICE,
                description="Export format",
                required=False,
                default="markdown",
                choices=["markdown", "json", "html"],
            ),
            CommandParameter(
                name="path",
                type=ParameterType.FILE,
                description="Output file path",
                required=False,
            ),
        ],
        examples=["/export", "/export json", "/export markdown ~/chat.md"],
    ),
    CommandDefinition(
        name="sessions",
        description="List all saved sessions",
        category=CommandCategory.SESSION,
        examples=["/sessions"],
    ),
    # Provider commands
    CommandDefinition(
        name="model",
        description="Switch to a different model",
        category=CommandCategory.PROVIDER,
        parameters=[
            CommandParameter(
                name="model",
                type=ParameterType.STRING,
                description="Model name or pattern",
                required=True,
            ),
        ],
        examples=["/model claude-3-opus", "/model gpt-4"],
    ),
    CommandDefinition(
        name="provider",
        description="Switch to a different provider",
        category=CommandCategory.PROVIDER,
        parameters=[
            CommandParameter(
                name="provider",
                type=ParameterType.CHOICE,
                description="Provider name",
                required=True,
                choices=["anthropic", "openai", "ollama", "google", "groq", "deepseek"],
            ),
        ],
        examples=["/provider ollama", "/provider anthropic"],
    ),
    CommandDefinition(
        name="providers",
        description="List available providers and their status",
        category=CommandCategory.PROVIDER,
        examples=["/providers"],
    ),
    CommandDefinition(
        name="models",
        description="List available models for current provider",
        category=CommandCategory.PROVIDER,
        examples=["/models"],
    ),
    # Mode commands
    CommandDefinition(
        name="mode",
        description="Switch agent mode",
        category=CommandCategory.MODE,
        parameters=[
            CommandParameter(
                name="mode",
                type=ParameterType.CHOICE,
                description="Agent mode",
                required=True,
                choices=["build", "plan", "review", "delegate", "explore", "chat"],
            ),
        ],
        examples=["/mode build", "/mode explore"],
    ),
    CommandDefinition(
        name="build",
        description="Switch to build mode (file operations enabled)",
        category=CommandCategory.MODE,
        aliases=["b"],
        examples=["/build"],
    ),
    CommandDefinition(
        name="plan",
        description="Switch to plan mode (read-only analysis)",
        category=CommandCategory.MODE,
        aliases=["p"],
        examples=["/plan"],
    ),
    CommandDefinition(
        name="review",
        description="Switch to review mode (diagnostics and findings)",
        category=CommandCategory.MODE,
        aliases=["r"],
        examples=["/review"],
    ),
    CommandDefinition(
        name="delegate",
        description="Switch to delegate mode (parallel work planning)",
        category=CommandCategory.MODE,
        aliases=["d"],
        examples=["/delegate"],
    ),
    CommandDefinition(
        name="explore",
        description="Switch to explore mode (codebase navigation)",
        category=CommandCategory.MODE,
        aliases=["e"],
        examples=["/explore"],
    ),
    # Tool commands
    CommandDefinition(
        name="tools",
        description="List available tools",
        category=CommandCategory.TOOL,
        parameters=[
            CommandParameter(
                name="category",
                type=ParameterType.STRING,
                description="Filter by category",
                required=False,
            ),
        ],
        examples=["/tools", "/tools search"],
    ),
    CommandDefinition(
        name="budget",
        description="Show or set tool budget",
        category=CommandCategory.TOOL,
        parameters=[
            CommandParameter(
                name="value",
                type=ParameterType.NUMBER,
                description="New budget value",
                required=False,
            ),
        ],
        examples=["/budget", "/budget 50"],
    ),
    # Context commands
    CommandDefinition(
        name="context",
        description="Show current context size and usage",
        category=CommandCategory.CONTEXT,
        examples=["/context"],
    ),
    CommandDefinition(
        name="compact",
        description="Compact the conversation context",
        category=CommandCategory.CONTEXT,
        parameters=[
            CommandParameter(
                name="target",
                type=ParameterType.NUMBER,
                description="Target context percentage (0-1)",
                required=False,
                default=0.5,
            ),
        ],
        examples=["/compact", "/compact 0.3"],
    ),
    CommandDefinition(
        name="add",
        description="Add file or directory to context",
        category=CommandCategory.CONTEXT,
        parameters=[
            CommandParameter(
                name="path",
                type=ParameterType.FILE,
                description="File or directory path",
                required=True,
            ),
        ],
        examples=["/add src/main.py", "/add ./tests"],
    ),
    # Help commands
    CommandDefinition(
        name="help",
        description="Show help for commands",
        category=CommandCategory.HELP,
        parameters=[
            CommandParameter(
                name="command",
                type=ParameterType.STRING,
                description="Command to get help for",
                required=False,
            ),
        ],
        aliases=["?"],
        examples=["/help", "/help model"],
    ),
    CommandDefinition(
        name="shortcuts",
        description="Show keyboard shortcuts",
        category=CommandCategory.HELP,
        examples=["/shortcuts"],
    ),
    # Debug commands
    CommandDefinition(
        name="debug",
        description="Toggle debug mode",
        category=CommandCategory.DEBUG,
        parameters=[
            CommandParameter(
                name="enabled",
                type=ParameterType.BOOLEAN,
                description="Enable or disable debug mode",
                required=False,
            ),
        ],
        hidden=True,
        examples=["/debug", "/debug true"],
    ),
    CommandDefinition(
        name="metrics",
        description="Show session metrics",
        category=CommandCategory.DEBUG,
        examples=["/metrics"],
    ),
    CommandDefinition(
        name="status",
        description="Show system status",
        category=CommandCategory.DEBUG,
        examples=["/status"],
    ),
]


class CommandRegistry:
    """Registry for managing slash commands.

    Provides lookup, validation, and execution of commands.
    """

    def __init__(self):
        """Initialize registry with built-in commands."""
        self._commands: Dict[str, CommandDefinition] = {}
        self._aliases: Dict[str, str] = {}
        self._handlers: Dict[str, Callable] = {}

        # Register built-in commands
        for cmd in SHARED_COMMANDS:
            self.register(cmd)

    def register(self, command: CommandDefinition) -> None:
        """Register a command definition."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def unregister(self, name: str) -> None:
        """Unregister a command."""
        if name in self._commands:
            cmd = self._commands[name]
            del self._commands[name]
            for alias in cmd.aliases:
                if alias in self._aliases:
                    del self._aliases[alias]

    def register_handler(
        self,
        name: str,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Register an execution handler for a command."""
        self._handlers[name] = handler

    def get(self, name: str) -> Optional[CommandDefinition]:
        """Get a command by name or alias."""
        if name in self._commands:
            return self._commands[name]
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        return None

    def list_commands(
        self,
        category: Optional[CommandCategory] = None,
        include_hidden: bool = False,
    ) -> List[CommandDefinition]:
        """List all commands, optionally filtered by category."""
        commands = list(self._commands.values())

        if category:
            commands = [c for c in commands if c.category == category]

        if not include_hidden:
            commands = [c for c in commands if not c.hidden]

        return sorted(commands, key=lambda c: (c.category.value, c.name))

    def parse_command(self, input_text: str) -> Optional[Dict[str, Any]]:
        """Parse a command string into name and arguments.

        Args:
            input_text: Command string like "/model claude-3-opus"

        Returns:
            Dict with 'name' and 'args' keys, or None if not a command
        """
        if not input_text.startswith("/"):
            return None

        parts = input_text[1:].split(maxsplit=1)
        if not parts:
            return None

        name = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        command = self.get(name)
        if not command:
            return {"name": name, "args": {}, "error": f"Unknown command: /{name}"}

        # Parse arguments
        args = self._parse_args(command, args_str)
        return {"name": command.name, "args": args}

    def _parse_args(
        self,
        command: CommandDefinition,
        args_str: str,
    ) -> Dict[str, Any]:
        """Parse argument string into dict."""
        args: Dict[str, Any] = {}

        if not command.parameters:
            return args

        # Simple positional parsing
        parts = args_str.split() if args_str else []

        for i, param in enumerate(command.parameters):
            if i < len(parts):
                value = parts[i]
                # Type conversion
                if param.type == ParameterType.NUMBER:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                elif param.type == ParameterType.BOOLEAN:
                    value = value.lower() in ("true", "1", "yes", "on")
                args[param.name] = value
            elif param.default is not None:
                args[param.name] = param.default

        return args

    async def execute(
        self,
        name: str,
        args: Dict[str, Any],
    ) -> Any:
        """Execute a command with the registered handler."""
        handler = self._handlers.get(name)
        if not handler:
            raise ValueError(f"No handler registered for command: {name}")

        return await handler(args)

    def to_dict(self) -> Dict[str, Any]:
        """Export all commands as a dictionary."""
        return {
            "commands": [cmd.to_dict() for cmd in self.list_commands(include_hidden=True)],
            "aliases": dict(self._aliases),
        }


# Global registry instance
command_registry = CommandRegistry()


# Decorator for registering command handlers
def command_handler(name: str):
    """Decorator to register a function as a command handler."""

    def decorator(func: Callable):
        command_registry.register_handler(name, func)
        return func

    return decorator
