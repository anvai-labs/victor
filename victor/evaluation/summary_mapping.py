# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared field and summary-mapping extraction for evaluation feedback helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

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


__all__ = ["extract_summary_mapping", "extract_value"]
