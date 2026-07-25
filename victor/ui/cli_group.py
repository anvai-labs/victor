# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Helpful command resolution for the Victor CLI.

Typer/Click default to a bare ``No such command 'X'.`` when a command name is
unknown, and only ever suggest matches among a group's *direct* children. That
hides genuinely available commands from users: ``victor vacuum`` fails even
though ``victor db vacuum`` exists, and typos like ``victor hwlp`` get no
guidance.

``SuggestingGroup`` overrides command resolution to fuzzy-match the attempted
name against both top-level commands and one level of nested subcommands, then
emits a ``Did you mean 'victor db vacuum'?`` hint.
"""

from __future__ import annotations

import difflib
from typing import Iterator, Optional, Tuple

import click
import typer

#: Minimum similarity for a candidate to be offered as a suggestion.
_SUGGEST_CUTOFF = 0.45
#: How many suggestions to show at most.
_MAX_SUGGESTIONS = 3


def _iter_command_paths(group: click.Group, ctx: click.Context) -> Iterator[Tuple[str, str]]:
    """Yield ``(match_key, display)`` pairs for a group and its nested subs.

    The ``match_key`` is what the user's input is fuzzy-matched against (a bare
    command name, or a ``"group sub"`` path). ``display`` is the full
    ``"<program> ..."`` string shown back to the user.
    """
    program = ctx.command_path or group.name or "victor"
    for name in sorted(group.list_commands(ctx)):
        cmd = group.get_command(ctx, name)
        if cmd is None:
            continue
        yield name, f"{program} {name}"
        if isinstance(cmd, click.Group):
            for sub in sorted(cmd.list_commands(ctx)):
                if sub == "help":
                    continue
                full = f"{program} {name} {sub}"
                # Match against both the bare sub-name and the full path so a
                # user typing just "vacuum" still hits "db vacuum".
                yield sub, full
                yield f"{name} {sub}", full


def suggest_command(ctx: click.Context, attempted: str, group: click.Group) -> Optional[str]:
    """Return a human-readable "did you mean?" message, or ``None``.

    ``None`` means nothing was close enough — callers should keep the original
    error. The message always starts with ``No such command '<attempted>'.`` so
    it can drop in for Click's default wording.
    """
    candidates = list(_iter_command_paths(group, ctx))
    keys = [key for key, _ in candidates]
    matches = difflib.get_close_matches(attempted, keys, n=_MAX_SUGGESTIONS, cutoff=_SUGGEST_CUTOFF)
    if not matches:
        return None

    displays: list[str] = []
    for key in matches:
        for candidate_key, display in candidates:
            if candidate_key == key and display not in displays:
                displays.append(display)
                break
    if not displays:
        return None

    if len(displays) == 1:
        return f"No such command '{attempted}'. Did you mean '{displays[0]}'?"

    bullet = "\n  ".join(displays)
    return f"No such command '{attempted}'. Did you mean one of:\n  {bullet}"


class SuggestingGroup(typer.core.TyperGroup):
    """Typer group that suggests nested commands when resolution fails.

    Only genuine unknown-command errors are enhanced; every other
    ``UsageError`` (missing arguments, bad options, etc.) is re-raised
    unchanged.

    Also owns the legacy-alias hint (UX P2): flat commands that moved under a
    verb group stay invocable but hidden; resolving one prints a one-line
    pointer to the new path on stderr. ``legacy_paths`` maps old top-level
    name -> new invocation, populated at registration time in ``cli.py``.
    """

    #: old top-level command name -> replacement invocation (e.g. "config profiles")
    legacy_paths: dict[str, str] = {}

    def resolve_command(self, ctx: click.Context, args: list[str]):
        attempted = args[0] if args else ""
        new_path = self.legacy_paths.get(attempted)
        if new_path:
            click.echo(
                f"note: 'victor {attempted}' is moving to 'victor {new_path}' "
                f"— the old form keeps working for now.",
                err=True,
            )
        try:
            return super().resolve_command(ctx, args)
        except click.exceptions.UsageError as exc:
            attempted = args[0] if args else ""
            # Only enhance true "command not found" failures: the attempted
            # name must actually be absent from this group.
            if attempted and self.get_command(ctx, attempted) is None:
                hint = suggest_command(ctx, attempted, self)
                if hint:
                    raise click.exceptions.UsageError(hint, ctx=ctx) from exc
            raise
