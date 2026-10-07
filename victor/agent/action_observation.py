# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").
"""Durable observations for one already-approved action; never an effect dispatcher.

A returned invocation is not a verified backend receipt. Pending and unknown actions
cannot be replayed or garbage-collected based on approval expiry.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from typing import Any, AsyncIterator, Iterator

from victor.framework.approval_binding import ActionObserver, digest


class ActionStateError(RuntimeError):
    """Durable action evidence is unavailable or conflicts with the claimed action."""


def begin_record(run: Any, binding: dict[str, Any]) -> dict[str, Any]:
    if (
        run.status != "resumed"
        or run.action is not None
        or not isinstance(binding, dict)
        or not run.session_id
        or binding.get("session_id") != run.session_id
        or not run.pending_tool
        or binding != run.pending_tool.get("binding")
        or binding.get("tool_name") != run.pending_tool.get("tool_name")
    ):
        raise ActionStateError("Action requires the original claimed approval and no prior intent")
    identity = {"version": 1, "run_id": run.run_id, "binding": binding}
    return {
        "version": 1,
        "action_id": digest(identity),
        "binding_digest": digest(binding),
        "state": "pending",
        "reported_success": None,
        "backend_receipt": None,
    }


def observe_record(
    action: dict[str, Any] | None, action_id: str, state: str, reported_success: bool | None
) -> dict[str, Any]:
    if (
        not isinstance(action, dict)
        or type(action.get("version")) is not int
        or action.get("version") != 1
        or action.get("action_id") != action_id
        or action.get("state") != "pending"
        or state not in {"returned", "unknown"}
        or (reported_success is not None and type(reported_success) is not bool)
        or (state == "unknown" and reported_success is not None)
    ):
        raise ActionStateError("Action observation conflicts with its pending intent")
    return {**action, "state": state, "reported_success": reported_success}


class ActionJournal:
    """One invocation's callbacks into the existing project paused-run store."""

    def __init__(self, store: Any, run_id: str, binding: dict[str, Any]) -> None:
        if getattr(store, "durable_actions", False) is not True:
            raise ActionStateError("Durable actions require persistent storage")
        self.store = store
        self.run_id = run_id
        self.binding = deepcopy(binding)
        self.action_id: str | None = None
        self.settlement_attempted = False
        self.return_observed = False
        self.reported_success: bool | None = None

    def begin(self) -> str:
        try:
            action = self.store.begin_action(self.run_id, self.binding)
            self.action_id = action["action_id"]
            return self.action_id
        except Exception as exc:
            raise ActionStateError(
                "Could not persist action intent; no dispatch authorized"
            ) from exc

    def returned(self, reported_success: bool | None) -> None:
        self.settlement_attempted = True
        try:
            self.store.observe_action(self.run_id, self.action_id, "returned", reported_success)
            self.return_observed = True
            self.reported_success = reported_success
        except Exception as exc:
            raise ActionStateError(
                "Action observation unavailable; reconciliation required"
            ) from exc

    def interrupted(self, original: BaseException) -> None:
        if self.action_id is None or self.settlement_attempted:
            return
        self.settlement_attempted = True
        try:
            self.store.observe_action(self.run_id, self.action_id, "unknown", None)
        except Exception:
            # Preserve cancellation/approval and leave durable intent unresolved.
            original.add_note("Action remains unresolved; observation persistence failed")

    async def begin_async(self) -> str:
        # A contended BEGIN IMMEDIATE can block up to the connection's 60 s
        # lock timeout; offload so a stalled write cannot freeze the event
        # loop that serves every other task.
        return await asyncio.to_thread(self.begin)

    async def returned_async(self, reported_success: bool | None) -> None:
        await asyncio.to_thread(self.returned, reported_success)

    async def interrupted_async(self, original: BaseException) -> None:
        # Cancellation may already be unwinding: shield the best-effort marker
        # so a second cancel cannot skip the durable "unknown" write. If the
        # shield itself is cancelled, the write continues detached and the
        # intent stays unresolved either way — the cancellation is delivered.
        await asyncio.shield(asyncio.to_thread(self.interrupted, original))


@contextmanager
def observe_dispatch(journal: ActionObserver | None) -> Iterator[str | None]:
    """Bracket the canonical executor call without providing a second dispatch path."""
    if journal is None:
        yield None
        return
    action_id = journal.begin()
    try:
        yield action_id
    except BaseException as exc:
        journal.interrupted(exc)
        raise


@asynccontextmanager
async def observe_dispatch_async(
    journal: ActionJournal | None,
) -> AsyncIterator[str | None]:
    """Async twin of observe_dispatch: intent persistence runs off the event loop.

    Ordering is unchanged — persist, recheck authority, dispatch — but a
    contended SQLite write now awaits in a worker thread instead of blocking
    the loop that serves every other task.
    """
    if journal is None:
        yield None
        return
    action_id = await journal.begin_async()
    try:
        yield action_id
    except BaseException as exc:
        await journal.interrupted_async(exc)
        raise
