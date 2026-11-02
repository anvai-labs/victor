"""Authored static-indexer controls; never imported or executed by the benchmark."""


class Registry:
    def get(self, key: str) -> int:
        return len(key)


class Cache:
    def get(self, key: str) -> int:
        return -len(key)


def shared_read(store: Registry, key: str) -> int:
    return store.get(key)


def shared_negative(store: Registry, key: str) -> int:
    return -store.get(key)
