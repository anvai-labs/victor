"""Canonical aggregation of retained attempts for one iterative member."""

from copy import deepcopy

from victor.teams.types import MemberResult


def aggregate_attempts(attempts: list[MemberResult], formation: str) -> MemberResult:
    """Sum independent attempt deltas and reject ambiguous session/accounting data."""
    if not attempts or len({item.member_id for item in attempts}) != 1:
        raise ValueError("Attempt aggregation requires exactly one nonempty member history")
    result = deepcopy(attempts[-1])
    failed = [item for item in attempts if not item.success]
    if failed:
        result.success = False
        result.error = failed[0].error or f"An earlier {formation} attempt failed"
    result.tool_calls_used = sum(item.tool_calls_used for item in attempts)
    result.duration_seconds = sum(item.duration_seconds for item in attempts)
    result.metadata[f"{formation}_attempts"] = [deepcopy(item.to_dict()) for item in attempts]
    usage: dict[str, int] = {}
    invalid = False
    session = attempts[0].metadata.get("session_id")
    for item in attempts:
        counters = item.metadata.get("usage")
        if (
            not isinstance(session, str)
            or not session
            or item.metadata.get("session_id") != session
            or not isinstance(counters, dict)
            or not {"input_tokens", "output_tokens", "total_tokens"} <= counters.keys()
            or any(type(value) is not int or value < 0 for value in counters.values())
            or counters["total_tokens"] != counters["input_tokens"] + counters["output_tokens"]
        ):
            invalid = True
            continue
        for key, value in counters.items():
            usage[key] = usage.get(key, 0) + value
    if invalid:
        result.success = False
        result.error = (
            f"{formation.capitalize()} member capture has missing or inconsistent session/usage"
        )
        result.metadata.pop("usage", None)
        result.metadata[f"{formation}_capture_error"] = True
    else:
        result.metadata["usage"] = usage
    return result
