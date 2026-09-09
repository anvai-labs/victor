# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Enumerated service capabilities consumed by the chat runtime.

FEP-0031 phase 1 is incremental: only task requirements have migrated so far.
The view freezes its bindings while preserving the session owner's live state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from victor.agent.session_state_accessor import SessionStateAccessor


class TaskRequirementState(Protocol):
    """The session state needed to prepare a turn's task requirements."""

    @property
    def required_files(self) -> list[str]: ...

    @required_files.setter
    def required_files(self, value: list[str]) -> None: ...

    @property
    def required_outputs(self) -> list[str]: ...

    @required_outputs.setter
    def required_outputs(self, value: list[str]) -> None: ...

    @property
    def read_files(self) -> set[str]: ...

    @property
    def all_files_read_nudge_sent(self) -> bool: ...

    @all_files_read_nudge_sent.setter
    def all_files_read_nudge_sent(self, value: bool) -> None: ...


class SessionTaskRequirementState:
    """Resolve state through the existing accessor after resets and restores."""

    __slots__ = ("_accessor",)

    def __init__(self, accessor: SessionStateAccessor) -> None:
        self._accessor = accessor

    @property
    def required_files(self) -> list[str]:
        return self._accessor.required_files

    @required_files.setter
    def required_files(self, value: list[str]) -> None:
        self._accessor.required_files = value

    @property
    def required_outputs(self) -> list[str]:
        return self._accessor.required_outputs

    @required_outputs.setter
    def required_outputs(self, value: list[str]) -> None:
        self._accessor.required_outputs = value

    @property
    def read_files(self) -> set[str]:
        return self._accessor.read_files_session

    @property
    def all_files_read_nudge_sent(self) -> bool:
        return self._accessor.all_files_read_nudge_sent

    @all_files_read_nudge_sent.setter
    def all_files_read_nudge_sent(self, value: bool) -> None:
        self._accessor.all_files_read_nudge_sent = value


@dataclass(frozen=True, slots=True)
class ChatRuntimeServices:
    """Explicit capabilities already migrated from the chat runtime facade."""

    session: TaskRequirementState
