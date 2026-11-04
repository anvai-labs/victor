from control_stores import Registry, shared_read as fetch


def compute(store: Registry, key: str) -> int:
    value = fetch(store, key)
    return value + 2
