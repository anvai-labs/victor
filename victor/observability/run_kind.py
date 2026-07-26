# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""What *kind* of run produced a session — recorded at emission, not guessed.

Downstream consumers need to tell a benchmark run from a delegate worker from a
person typing, because the three deserve different treatment: an evaluation
carries a ground-truth verdict, a delegate worker runs under an artificial turn
budget, and interactive work has no verdict at all.

Until now that had to be inferred from prompt text, and the inference was wrong.
The turn-budget notice ("WARNING: N turns remaining out of 10") is emitted by
the shared agentic loop, so classifying on it counted delegate work as benchmark
runs and overstated the eval share of trace evidence by roughly 2x (see
``docs/analysis/2026-07-25-prompt-evolution-audit.md``). No prompt string can fix
that, because the prompt genuinely is shared. The run kind has to be recorded by
whoever *starts* the run.

The current kind lives in a context variable, so nested scopes restore correctly
and concurrent tasks in the same process do not clobber each other. An
environment variable seeds the default, which lets a subprocess-based runner tag
its children without threading arguments through.
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Coroutine, Final, Iterator, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

RUN_KIND_ENV: Final = "VICTOR_RUN_KIND"


class RunKind:
    """Canonical run kinds. String constants so they serialize as themselves."""

    INTERACTIVE: Final = "interactive"
    EVALUATION: Final = "evaluation"
    DELEGATE: Final = "delegate"
    HEADLESS: Final = "headless"

    ALL: Final = frozenset({INTERACTIVE, EVALUATION, DELEGATE, HEADLESS})


_run_kind: ContextVar[str] = ContextVar("victor_run_kind", default="")


def _from_env() -> str:
    """Seed value from the environment, ignoring anything unrecognized."""
    raw = os.environ.get(RUN_KIND_ENV, "").strip().lower()
    if raw in RunKind.ALL:
        return raw
    if _headless_configured():
        return RunKind.HEADLESS
    return RunKind.INTERACTIVE


def _headless_configured() -> bool:
    return os.environ.get("VICTOR_HEADLESS_MODE", "").strip().lower() in {"1", "true", "yes"}


def current_run_kind() -> str:
    """The kind of run in progress on this call stack.

    Falls back to the environment (then to headless/interactive) so a process
    that never enters a scope still emits something meaningful rather than
    leaving consumers to guess.
    """
    return _run_kind.get() or _from_env()


@contextmanager
def run_kind_scope(kind: str) -> Iterator[str]:
    """Mark everything emitted inside this block as ``kind``.

    Nesting is honoured: a delegate spawned inside an evaluation is tagged
    ``delegate`` for its duration and the evaluation tag resumes afterwards, so
    the innermost owner of the work is the one that names it.
    """
    if kind not in RunKind.ALL:
        raise ValueError(f"Unknown run kind {kind!r}; expected one of {sorted(RunKind.ALL)}")
    token = _run_kind.set(kind)
    try:
        yield kind
    finally:
        _run_kind.reset(token)


def tagged_run(kind: str) -> Callable[[F], F]:
    """Decorate an async entry point so its whole subtree is tagged ``kind``.

    Equivalent to wrapping the body in ``run_kind_scope``, but without
    re-indenting an existing function — which matters for entry points whose
    bodies are large ``try``/``async with`` blocks where a wrapper would produce
    a diff that obscures the one-line change.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> object:
            with run_kind_scope(kind):
                return await func(*args, **kwargs)

        return cast(F, wrapper)

    return decorate


__all__ = [
    "RUN_KIND_ENV",
    "RunKind",
    "current_run_kind",
    "run_kind_scope",
    "tagged_run",
]
