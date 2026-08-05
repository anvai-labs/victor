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

"""Plugin protocol for Victor.

This module defines the protocol that all Victor plugins must implement
to support dynamic registration of tools, verticals, commands, and strategies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import typer
from victor_contracts import PluginContext, VictorPlugin as SdkVictorPlugin


@runtime_checkable
class VictorPlugin(SdkVictorPlugin, Protocol):
    """Protocol for all Victor plugins.

    Plugins are discovered via the 'victor.plugins' entry point.
    """

    @property
    def name(self) -> str:
        """Return the plugin identifier (e.g., 'coding', 'rag')."""
        ...

    def register(self, context: PluginContext) -> None:
        """Register services, tools, and verticals into the container/registries.

        This is called during application bootstrap.
        """
        ...

    def get_cli_app(self) -> Optional[typer.Typer]:
        """Return a Typer app to be registered as a subcommand.

        If returned, this app will be registered under the plugin's name.

        .. deprecated::
            Use context.register_command() inside register() instead.
        """
        ...

    def on_activate(self) -> None:
        """Called when this plugin's vertical is activated."""
        ...

    def on_deactivate(self) -> None:
        """Called when this plugin's vertical is being deactivated."""
        ...

    def health_check(self) -> Dict[str, Any]:
        """Return health status for this plugin."""
        ...
