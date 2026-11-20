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

"""The one sandboxed-launch seam: argv AND owned descriptors, plus cleanup.

Backends that wrap a host command may open descriptors the wrapped process
must inherit — the Linux bwrap backend opens a seccomp filter descriptor
that ``bwrap --seccomp FD`` reads after exec. Forwarding the wrapped argv
without the descriptors leaves the child without its filter (or refuses to
start outright), and never draining them leaks one open descriptor in the
parent per launch.

Every spawn site — the canonical subprocess executor and the direct
terminal/git/search routes — goes through :func:`prepare_launch` followed by
:func:`run_launch` / :func:`run_launch_async`, so the argv and the
``pass_fds`` contract travel together and the owned descriptors are closed
on every exit path, including spawn failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from victor.tools.sandbox import SandboxBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxLaunch:
    """A spawn unit: the argv to run plus the descriptors it must inherit.

    ``argv`` is a plain (unwrapped) argv when no enforcing backend applies;
    otherwise it is the backend-wrapped argv and ``pass_fds`` carries the
    backend-owned descriptors the child must inherit (bwrap's seccomp
    filter). ``command`` may be a shell string when the caller is a shell
    surface; the spawn helpers forward it verbatim alongside ``shell=True``.
    """

    argv: Union[str, List[str]]
    pass_fds: Tuple[int, ...] = ()
    sandbox: Optional["SandboxBackend"] = None


def prepare_launch(
    sandbox: Optional["SandboxBackend"],
    argv: Union[str, List[str]],
    cwd: Optional[Union[str, Path]] = None,
) -> SandboxLaunch:
    """Wrap ``argv`` with ``sandbox`` and collect its owned descriptors.

    A ``None`` or no-op backend returns the argv unchanged with no
    descriptors (historical raw exec under a non-enforcing policy). An
    enforcing backend wraps the argv and DRAINS its pending descriptors —
    the invariant is that every ``wrap_argv`` is followed by a
    ``take_pass_fds``, so a launch can never leak a backend descriptor even
    when the caller abandons the launch before spawning.
    """
    if sandbox is None or sandbox.type_name == "none":
        return SandboxLaunch(argv=list(argv) if isinstance(argv, list) else argv, pass_fds=())
    if isinstance(argv, str):
        # Enforcing backends wrap argv lists only; shell surfaces hand the
        # executor a ['/bin/sh', '-c', cmd] list before wrapping.
        raise TypeError("an enforcing sandbox wraps argv lists, not shell strings")
    cwd_path = Path(cwd) if cwd is not None else None
    wrapped = sandbox.wrap_argv(list(argv), cwd_path)
    return SandboxLaunch(argv=wrapped, pass_fds=tuple(sandbox.take_pass_fds()), sandbox=sandbox)


def close_pass_fds(pass_fds: Sequence[int]) -> None:
    """Close backend descriptors whose inheritance duty ended after spawn."""
    for fd in pass_fds:
        try:
            os.close(fd)
        except OSError:
            pass


def run_launch(launch: SandboxLaunch, **run_kwargs: Any) -> "subprocess.CompletedProcess[Any]":
    """Blocking spawn of :class:`SandboxLaunch` with descriptor cleanup.

    Forwards ``pass_fds`` to :func:`subprocess.run` and closes the owned
    descriptors on every exit, including a raised spawn failure — repeated
    failing launches leave the parent descriptor count stable.
    """
    owned = launch.pass_fds
    try:
        return subprocess.run(  # nosec B603 - argv is backend-wrapped or caller-supplied
            launch.argv,
            pass_fds=owned,
            **run_kwargs,
        )
    finally:
        close_pass_fds(owned)


async def run_launch_async(
    launch: SandboxLaunch, **exec_kwargs: Any
) -> "asyncio.subprocess.Process":
    """Async spawn of :class:`SandboxLaunch` with descriptor cleanup.

    Forwards ``pass_fds`` to :func:`asyncio.create_subprocess_exec` and
    closes the owned descriptors once the child is running (the child holds
    its inherited copy; bwrap reads the filter after exec).
    """
    owned = launch.pass_fds
    try:
        argv = launch.argv
        if isinstance(argv, str):
            raise TypeError("run_launch_async requires an argv list, not a shell string")
        return await asyncio.create_subprocess_exec(
            *argv,
            pass_fds=owned,
            **exec_kwargs,
        )
    finally:
        close_pass_fds(owned)


__all__ = [
    "SandboxLaunch",
    "prepare_launch",
    "close_pass_fds",
    "run_launch",
    "run_launch_async",
]
