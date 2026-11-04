from control_stores import Registry, shared_negative as lookup


def compute(store: Registry, key: str) -> int:
    value = lookup(store, key)
    return value + 2
