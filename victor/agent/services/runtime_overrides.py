"""Apply temporary turn overrides together and restore every changed owner."""

from typing import Any, Callable

_MISSING = object()
Change = tuple[str, Callable[[], Any], Callable[[Any], None], Any]


class OverrideRestorationError(RuntimeError):
    """An owner could not restore its state; the runtime must be recreated."""


def override_failed(owner: Any) -> bool:
    """Read the sticky failed-restore marker without coupling callers to its field."""
    return getattr(owner, "_runtime_override_error", False) is True


def mark_override_failed(owner: Any) -> None:
    """Mark an owner as requiring recreation after failed restoration."""
    setattr(owner, "_runtime_override_error", True)


def runtime_orchestrator(owner: Any) -> Any:
    """Resolve the legacy owner link at the composition boundary."""
    return getattr(owner, "_orchestrator")


def attribute_change(
    name: str, owner: Any, attribute: str, value: Any, *, state: dict[str, Any] | None = None
) -> Change:
    """Describe an attribute update, preserving an absent original attribute."""

    def restore_or_set(previous: Any) -> None:
        if previous is _MISSING:
            if hasattr(owner, attribute):
                delattr(owner, attribute)
        else:
            setattr(owner, attribute, previous)

    getter = (
        (lambda: state.get(attribute, _MISSING))
        if state is not None
        else (lambda: getattr(owner, attribute, _MISSING))
    )
    return name, getter, restore_or_set, value


def budget_changes(orchestrator: Any, fallback: Any, budget: int) -> list[Change]:
    """Collect existing budget owners without changing any of them."""
    changes: list[Change] = []
    budget = max(0, budget)
    if orchestrator is not None and hasattr(orchestrator, "tool_budget"):
        changes.append(
            attribute_change("orchestrator_tool_budget", orchestrator, "tool_budget", budget)
        )
    coordinator = getattr(orchestrator, "task_coordinator", None)
    if coordinator is not None and hasattr(coordinator, "tool_budget"):
        changes.append(
            attribute_change("task_coordinator_tool_budget", coordinator, "tool_budget", budget)
        )
    service = getattr(orchestrator, "_tool_service", None)
    if service is None:
        service = getattr(fallback, "_tool_service", None)
    if service is not None and hasattr(service, "get_tool_budget"):
        if not callable(getattr(service, "set_tool_budget", None)):
            raise RuntimeError("Tool service budget cannot be overridden")

        def get_budget() -> Any:
            if hasattr(service, "budget"):
                return service.budget
            if hasattr(service, "get_budget_info"):
                return service.get_budget_info()["max"]
            return service.get_tool_budget()

        changes.append(("tool_service_budget", get_budget, service.set_tool_budget, budget))
    pipeline = getattr(orchestrator, "_tool_pipeline", None)
    if pipeline is None:
        pipeline = getattr(fallback, "_tool_pipeline", None)
    config = getattr(pipeline, "config", None)
    if config is not None and hasattr(config, "tool_budget"):
        changes.append(attribute_change("pipeline_tool_budget", config, "tool_budget", budget))
    return changes


def apply_overrides(changes: list[Change]) -> dict[str, Any]:
    """Snapshot before mutation; a failed setter triggers rollback and an error."""
    snapshot = {name: getter() for name, getter, _, _ in changes}
    undo: list[tuple[Callable[[Any], None], Any]] = []
    snapshot["_restore_actions"] = undo
    try:
        for name, _, setter, value in changes:
            # Include a setter that mutates and then raises in the rollback.
            undo.append((setter, snapshot[name]))
            setter(value)
    except Exception:
        restore_overrides(snapshot)
        raise
    return snapshot


def restore_overrides(snapshot: dict[str, Any]) -> None:
    """Attempt every restoration even when one owner fails."""
    errors: list[Exception] = []
    for setter, previous in reversed(snapshot.pop("_restore_actions", [])):
        try:
            setter(previous)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise OverrideRestorationError(
            "Runtime override restoration failed; recreate the session"
        ) from ExceptionGroup("Failed restorations", errors)
