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

"""Command Provider Protocol.

This protocol defines the interface that vertical packages must implement
to register their CLI commands with the Victor framework.

Verticals register CLI commands via the `victor.commands` entry point group.
Each registration function should conform to this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import typer


@runtime_checkable
class CommandProvider(Protocol):
    """Protocol for CLI command registration functions.

    Command registration functions are called by the framework to register
    vertical-specific CLI commands. These functions are registered via the
    `victor.commands` entry point group.

    The function signature must be:
        def register_commands(app: typer.Typer) -> None

    Example:
        # In victor_coding/ui/commands/check_codebase_index.py:
        def register_commands(app: typer.Typer) -> None:
            \"\"\"Register coding-specific CLI commands.\"\"\"
            @app.command()
            def check_codebase_index(
                path: str = ".",
                verbose: bool = False,
            ):
                \"\"\"Check the codebase index status.\"\"\"
                # This import runs inside the external vertical package
                from my_vertical.codebase.indexer import CodebaseIndex
                index = CodebaseIndex(path)
                status = index.check_status()
                if verbose:
                    print(status.details)
                else:
                    print(status.summary)

        # In victor-coding/pyproject.toml:
        [project.entry-points."victor.commands"]
        check_codebase_index = "victor_coding.ui.commands.check_codebase_index:register_commands"

        # Framework usage:
        from importlib.metadata import entry_points
        app = typer.Typer()
        for ep in entry_points(group="victor.commands"):
            register_func = ep.load()
            register_func(app)
    """

    def __call__(self, app: typer.Typer) -> None:
        """Register CLI commands with the given Typer app.

        Args:
            app: The Typer application to register commands with.
        """
        ...
