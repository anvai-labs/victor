# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").
"""Invocation observations and backend receipts; never an effect dispatcher.

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


def begin_record(
    run: Any, binding: dict[str, Any], recovery: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    from victor.framework.action_recovery import RecoveryIdentity

    if recovery is not None:
        recovery = RecoveryIdentity.from_dict(recovery).to_dict()
    identity = {"version": 1, "run_id": run.run_id, "binding": binding}
    return {
        "version": 2 if recovery is not None else 1,
        **({"recovery": recovery} if recovery is not None else {}),
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
        or action.get("version") not in (1, 2)
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
        self.recovery: dict[str, Any] | None = None
        self.lookup_request: Any = None

    def bind_recovery(self, identity: dict[str, Any] | None, contract: str) -> None:
        if self.action_id is not None or contract != self.binding["contract"]:
            raise ActionStateError("Recovery requires the original approved tool contract")
        self.recovery = deepcopy(identity)

    def begin(self) -> str:
        try:
            from victor.framework.action_recovery import ActionLookup, RecoveryIdentity

            action = self.store.begin_action(
                self.run_id,
                self.binding,
                **({"recovery": self.recovery} if self.recovery is not None else {}),
            )
            self.action_id = action["action_id"]
            if self.recovery is not None:
                self.lookup_request = ActionLookup(
                    action["action_id"],
                    action["binding_digest"],
                    RecoveryIdentity.from_dict(action["recovery"]),
                )
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
            # A lost race with a committed returned() write is not "unresolved":
            # only flag when the action is genuinely still pending.
            try:
                settled = self.store.get(self.run_id)
            except Exception:
                settled = None
            if settled is not None and (settled.action or {}).get("state") == "returned":
                return
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


def _lookup_request(run: Any) -> Any:
    from victor.framework.action_recovery import ActionLookup, RecoveryIdentity

    if run is None:
        raise ActionStateError("Persisted action is unavailable")
    action = run.action
    binding = (run.pending_tool or {}).get("binding")
    if (
        run.status != "resumed"
        or not isinstance(action, dict)
        or type(action.get("version")) is not int
        or action["version"] != 2
        or action.get("state") not in {"pending", "unknown", "returned"}
        or not isinstance(binding, dict)
        or binding.get("session_id") != run.session_id
        or binding.get("tool_name") != run.pending_tool.get("tool_name")
        or ("agent_id" in binding and binding["agent_id"] != run.agent_id)
        or action.get("binding_digest") != digest(binding)
    ):
        raise ActionStateError("Recovery requires the original persisted action binding")
    try:
        return ActionLookup(
            action["action_id"],
            action["binding_digest"],
            RecoveryIdentity.from_dict(action["recovery"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionStateError("Malformed persisted recovery identity") from exc


def _receipt_record(request: Any, receipt: Any) -> dict[str, Any]:
    from victor.framework.action_recovery import ActionLookup, BackendReceipt

    if type(request) is not ActionLookup or type(receipt) is not BackendReceipt:
        raise ActionStateError("A typed authoritative backend receipt is required")
    try:
        record = receipt.to_dict()
        if (receipt.action_id, receipt.binding_digest, receipt.identity) != (
            request.action_id,
            request.binding_digest,
            request.identity,
        ):
            raise ValueError("Receipt identity mismatch")
    except (TypeError, ValueError) as exc:
        raise ActionStateError("Backend receipt does not match the original action") from exc
    return record


def retain_receipt_record(run: Any, request: Any, receipt: Any) -> dict[str, Any]:
    """Pure compare-and-set rule used inside the existing store transaction."""
    if _lookup_request(run) != request:
        raise ActionStateError("Persisted recovery binding changed")
    record = _receipt_record(request, receipt)
    existing = run.action.get("backend_receipt")
    if existing is not None and existing != record:
        raise ActionStateError("Conflicting backend receipt; reconciliation required")
    return {**run.action, "backend_receipt": record}


def require_receipt_access(run: Any, runtime: Any) -> tuple[Any, Any]:
    """Authorize receipt metadata through the existing runtime/registry/RBAC owner.

    This is embedded configured authority, not a hosted authenticated principal.
    It grants no tool execution, approval, result publication or continuation.
    """
    from victor.framework.action_recovery import BackendReceipt, recovery_identity
    from victor.framework.approval_binding import tool_contract

    try:
        request = _lookup_request(run)
        binding = run.pending_tool["binding"]
        executor = runtime._tool_pipeline.executor
        tool = executor.tools.get(binding["tool_name"])
        if (
            not run.session_id
            or runtime.active_session_id != run.session_id
            or digest(executor.current_user) != binding["authority"]
            or tool_contract(tool) != binding["contract"]
            or recovery_identity(tool) != request.identity.to_dict()
            or not executor._check_rbac(tool, tool.name)[0]
        ):
            raise PermissionError("Receipt access requires original session, authority and backend")
        if run.action.get("backend_receipt") is not None:
            _receipt_record(request, BackendReceipt.from_dict(run.action["backend_receipt"]))
        return tool, tool.action_recovery
    except PermissionError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise PermissionError("Receipt access context unavailable or changed") from exc


async def reconcile_action_receipt(
    store: Any, run_id: str, runtime: Any, *, timeout_seconds: float
) -> dict[str, Any]:
    """Read backend evidence and persist it; never execute or continue an action."""
    import math
    from victor.agent.paused_run_store import ProjectDbPausedRunStore
    from victor.framework.action_recovery import BackendReceipt

    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 60
    ):
        raise ValueError("Receipt lookup timeout must be finite, positive and at most 60 seconds")
    if not isinstance(store, ProjectDbPausedRunStore):
        raise ValueError("Receipt reconciliation requires project database storage")
    run = await asyncio.to_thread(store.get, run_id)
    if run is None:
        raise ValueError("Unknown paused run")
    if not run.session_id or getattr(runtime, "active_session_id", None) != run.session_id:
        raise PermissionError("Receipt access requires the restored original session")
    if run.action is None or (
        type(run.action.get("version")) is int
        and run.action["version"] == 1
        and "recovery" not in run.action
        and run.action.get("backend_receipt") is None
    ):
        return {"version": 1, "run_id": run_id, "status": "unsupported", "action": run.action}
    request = _lookup_request(run)
    tool, capability = require_receipt_access(run, runtime)

    def authorize(current: Any) -> None:
        if _lookup_request(current) != request:
            raise ActionStateError("Persisted recovery binding changed")
        current_tool, current_capability = require_receipt_access(current, runtime)
        if current_tool is not tool or current_capability is not capability:
            raise PermissionError("Configured recovery capability changed")

    def result(current: Any, reason: str | None = None) -> dict[str, Any]:
        authorize(current)
        recorded = current.action.get("backend_receipt")
        if recorded is not None:
            _receipt_record(request, BackendReceipt.from_dict(recorded))
        return {
            "version": 1,
            "run_id": run_id,
            "status": "verified" if recorded is not None else "unknown",
            "reason": None if recorded is not None else reason,
            "action": current.action,
        }

    if run.action.get("backend_receipt") is not None:
        return result(run)
    receipt = None
    reason = "receipt_absent"
    try:
        # Adapters must cooperate with cancellation; no hard bound on hostile code.
        receipt = await asyncio.wait_for(capability.lookup(request), timeout=timeout_seconds)
    except TimeoutError:
        reason = "lookup_timeout"
    except Exception:
        reason = "lookup_failed"  # Do not disclose credential-bearing adapter errors.
    if receipt is not None:
        _receipt_record(request, receipt)
    current = await asyncio.to_thread(store.get, run_id)
    authorize(current)
    if receipt is None:
        return result(current, reason)
    action = await asyncio.to_thread(store.retain_receipt, run_id, request, receipt)
    current.action = action
    return result(current)
