from control_stores import Registry, shared_read as lookup


def compute(store: Registry, key: str) -> int:
    value = lookup(store, key)
    return value + 2
