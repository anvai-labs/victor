# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared field and summary-mapping extraction for evaluation feedback helpers."""

from __future__ import annotations

from typing import Any, Callable, Hashable, Mapping, Optional, Sequence, Tuple

_SUMMARY_CONTAINERS = ("metadata", "failure_details", "completion_signals")


def extract_value(value: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping, or the ``key`` attribute of any other object.

    Args:
        value: A mapping or an arbitrary object.
        key: Mapping key or attribute name.
        default: Returned when the key/attribute is absent.
    """
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_summary_mapping(value: Any, key: str) -> Optional[dict[str, Any]]:
    """Find a summary mapping stored under ``key`` on a task, trace or payload.

    Lookup order: ``value[key]`` (or ``value.key``), then ``key`` inside the
    ``metadata``, ``failure_details`` and ``completion_signals`` containers,
    then the same search on ``value.trace`` (recursively, unless the trace is
    ``value`` itself). Only mapping summaries count.

    Args:
        value: A mapping or object that may carry the summary.
        key: Summary key, e.g. ``"topology_summary"``.

    Returns:
        A shallow ``dict`` copy of the first summary found, or ``None``.
    """
    if isinstance(value, Mapping):
        summary = value.get(key)
        if isinstance(summary, Mapping):
            return dict(summary)
    else:
        summary = getattr(value, key, None)
        if isinstance(summary, Mapping):
            return dict(summary)
    for container_name in _SUMMARY_CONTAINERS:
        container = extract_value(value, container_name)
        if isinstance(container, Mapping):
            summary = container.get(key)
            if isinstance(summary, Mapping):
                return dict(summary)
    trace = extract_value(value, "trace")
    if trace is not None and trace is not value:
        return extract_summary_mapping(trace, key)
    return None


def extract_sequence(value: Any, key: str) -> list[Any]:
    """Return ``key`` (mapping key or attribute) as a list when it is a sequence, else ``[]``."""
    raw_value = extract_value(value, key, [])
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    return []


def as_event_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    """Return ``value`` (or its ``to_dict()``) when it is a mapping, else None."""
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return value if isinstance(value, Mapping) else None


def coerce_optional_text(value: Any) -> Optional[str]:
    """Stripped text of ``value`` (unwrapping enum ``.value``); None when blank."""
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def coerce_optional_float(value: Any) -> Optional[float]:
    """``float(value)``; None for None, ``""`` or unparseable input."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_int(value: Any) -> Optional[int]:
    """``int(value)``; None for None, ``""`` or unparseable input."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


EventNormalizer = Callable[[Any], Optional[dict[str, Any]]]


def collect_events(
    value: Any,
    sources: Sequence[Tuple[str, EventNormalizer]],
    dedup_key: Callable[[dict[str, Any]], Hashable],
) -> list[dict[str, Any]]:
    """Collect normalized events from a task, trace or payload object.

    For each ``(key, normalize)`` source, events are read from ``value`` and
    then from each summary container (``metadata``, ``failure_details``,
    ``completion_signals``), and the same walk recurses into ``value.trace``.
    Normalizers return None to drop an item. Events are de-duplicated by
    ``dedup_key``, keeping the first occurrence in walk order.
    """
    events: list[dict[str, Any]] = []

    def _append(items: Any, normalize: EventNormalizer) -> None:
        for item in items:
            normalized = normalize(item)
            if normalized is not None:
                events.append(normalized)

    def _walk(node: Any) -> None:
        for key, normalize in sources:
            _append(extract_sequence(node, key), normalize)
        for container_name in _SUMMARY_CONTAINERS:
            container = extract_value(node, container_name)
            if isinstance(container, Mapping):
                for key, normalize in sources:
                    _append(extract_sequence(container, key), normalize)
        trace = extract_value(node, "trace")
        if trace is not None and trace is not node:
            _walk(trace)

    _walk(value)

    deduped: list[dict[str, Any]] = []
    seen: set[Hashable] = set()
    for event in events:
        key = dedup_key(event)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


__all__ = [
    "as_event_mapping",
    "collect_events",
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_optional_text",
    "extract_sequence",
    "extract_summary_mapping",
    "extract_value",
]
