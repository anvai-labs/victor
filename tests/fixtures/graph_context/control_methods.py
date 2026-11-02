"""Authored call-target controls; source inspection supplies the narrow oracle."""

from typing import Any

from control_stores import Cache, Registry


def registry_read(store: Registry, key: str) -> int:
    value = store.get(key)
    return value + 2


def renamed_read(receiver: Registry, name: str) -> int:
    result = receiver.get(name)
    return result + 2


def cache_read(store: Cache, key: str) -> int:
    value = store.get(key)
    return value + 2


def changed_operator(store: Registry, key: str) -> int:
    value = store.get(key)
    return value - 2


def changed_literal(store: Registry, key: str) -> int:
    value = store.get(key)
    return value + 3


def unknown_first(store: Any, key: str) -> int:
    value = store.get(key)
    return value + 2


def unknown_second(store: Any, key: str) -> int:
    value = store.get(key)
    return value + 2


def no_call_first(value: int) -> int:
    return value + 2


def no_call_second(item: int) -> int:
    return item + 2
