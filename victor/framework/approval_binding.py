# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0.
"""Internal exact-action capability for FEP-0029 single-call durable resume.

These context variables are runtime-owned, never model/tool arguments. A grant
only satisfies its original ASK; normal policy and executor checks still run.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable


class ApprovalBindingError(PermissionError):
    """Approval no longer authorizes this exact execution."""


def parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):

        def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ApprovalBindingError("Duplicate approval argument key")
                result[key] = item
            return result

        try:
            value = json.loads(value, object_pairs_hook=unique_pairs)
        except (ValueError, TypeError) as exc:
            raise ApprovalBindingError("Invalid approval arguments") from exc
    if not isinstance(value, dict):
        raise ApprovalBindingError("Approval arguments must be an object")
    digest(value)  # Also reject NaN, non-JSON objects and non-string object keys.
    return value


def digest(value: Any) -> str:
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ApprovalBindingError("Approval object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ApprovalBindingError("Approval value is not JSON")

    validate(value)
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ApprovalBindingError("Invalid approval payload") from exc
    return hashlib.sha256(encoded.encode()).hexdigest()


def proposal(call: dict[str, Any]) -> dict[str, Any]:
    call_id = call.get("id") or call.get("tool_call_id")
    name = call.get("name") or (call.get("function") or {}).get("name")
    args = (
        call.get("arguments")
        if "arguments" in call
        else (call.get("function") or {}).get("arguments")
    )
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        raise ApprovalBindingError("Approval requires an exact call ID and tool name")
    return {"id": call_id, "name": name, "arguments": deepcopy(parse_arguments(args))}


def tool_contract(tool: Any) -> str:
    return digest(
        {
            "name": tool.name,
            "parameters": tool.parameters,
            "access_mode": getattr(getattr(tool, "access_mode", None), "value", None),
        }
    )


@dataclass
class ApprovalCall:
    source: dict[str, Any]
    contract: str = ""
    authority: str = ""


current_approval_call: ContextVar[ApprovalCall | None] = ContextVar("approval_call", default=None)


def request_binding(
    tool_name: str, arguments: dict[str, Any], policy: str, scope: dict[str, Any]
) -> dict[str, Any] | None:
    call = current_approval_call.get()
    if call is None:
        return None  # Non-durable paths keep their existing contract.
    if not call.contract:
        raise ApprovalBindingError("Approval tool contract is unavailable")
    return {
        "version": 1,
        "call_id": call.source["id"],
        "proposal": digest(call.source),
        "tool_name": tool_name,
        "payload": digest(arguments),
        "contract": call.contract,
        "policy": policy,
        "scope": digest(scope),
        "authority": call.authority,
    }


@dataclass
class ApprovalGrant:
    binding: dict[str, Any]
    expires_at: float
    session_is_current: Callable[[], bool]
    policy_checked: bool = False
    ask_consumed: bool = False
    dispatched: bool = False

    def check(self, tool_name: str, arguments: dict[str, Any]) -> None:
        call = current_approval_call.get()
        if (
            not self.session_is_current()
            or self.dispatched
            or not math.isfinite(self.expires_at)
            or time.time() >= self.expires_at
            or call is None
            or call.source["id"] != self.binding["call_id"]
            or digest(call.source) != self.binding["proposal"]
            or call.contract != self.binding["contract"]
            or call.authority != self.binding["authority"]
            or tool_name != self.binding["tool_name"]
            or digest(arguments) != self.binding["payload"]
        ):
            raise ApprovalBindingError(
                "Approval changed, expired or already consumed; request new approval"
            )

    def check_policy(
        self, tool_name: str, arguments: dict[str, Any], scope: dict[str, Any]
    ) -> None:
        self.check(tool_name, arguments)
        if digest(scope) != self.binding["scope"]:
            raise ApprovalBindingError("Approval policy scope changed")
        self.policy_checked = True

    def approve_ask(self, policy: str) -> bool:
        if self.ask_consumed or policy != self.binding["policy"]:
            raise ApprovalBindingError("Approval does not cover this ASK")
        self.ask_consumed = True
        return True

    def dispatch(self, tool: Any, arguments: dict[str, Any], authority: Any) -> None:
        self.check(tool.name, arguments)
        if (
            not self.policy_checked
            or tool_contract(tool) != self.binding["contract"]
            or digest(authority) != self.binding["authority"]
        ):
            raise ApprovalBindingError(
                "Approval requires current policy and unchanged tool contract"
            )
        self.dispatched = True  # Consume before any possible effect, never reopen on failure.


current_approval_grant: ContextVar[ApprovalGrant | None] = ContextVar(
    "approval_grant", default=None
)
