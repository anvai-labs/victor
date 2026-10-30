# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for victor.framework.namespaced_registry (shared flat-registry base)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional

import pytest

from victor.framework.chain_registry import ChainRegistry
from victor.framework.namespaced_registry import (
    NamespacedRegistry,
    make_key,
    split_qualified_name,
)
from victor.framework.persona_registry import PersonaRegistry

_test_logger = logging.getLogger("tests.namespaced_registry")


@dataclass
class _Item:
    """Simple indexed value used by the toy registry."""

    colour: str
    tags: List[str] = field(default_factory=list)
    vertical: Optional[str] = None


class _ToyRegistry(NamespacedRegistry[_Item]):
    """Minimal concrete registry exercising the base hooks."""

    _KIND = "Toy"
    _KIND_PLURAL = "toys"
    _INDEX_NAMES = ("colour", "tag")
    _logger = _test_logger

    def __init__(self) -> None:
        super().__init__()
        self.forgotten: List[str] = []
        self.extra_cleared = 0

    def _index_terms(self, source: Any) -> Mapping[str, Iterable[str]]:
        item: _Item = source
        return {"colour": [item.colour], "tag": item.tags}

    def _forget(self, key: str) -> None:
        self.forgotten.append(key)

    def _clear_extra(self) -> None:
        self.extra_cleared += 1

    def register(
        self, name: str, item: _Item, vertical: Optional[str] = None, replace: bool = False
    ) -> bool:
        return self._put(make_key(name, vertical), item, replace=replace, index_source=item)

    def register_from_vertical(self, vertical_name: str, items: dict, replace: bool = True) -> int:
        return self._register_from_vertical(
            vertical_name,
            items,
            lambda n, i: self.register(n, i, vertical=vertical_name, replace=replace),
        )


class _OtherToyRegistry(_ToyRegistry):
    """Second subclass to prove singleton separation down the hierarchy."""


@pytest.fixture(autouse=True)
def _reset_singletons():
    for cls in (_ToyRegistry, _OtherToyRegistry, ChainRegistry, PersonaRegistry):
        cls.reset_instance()
    yield
    for cls in (_ToyRegistry, _OtherToyRegistry, ChainRegistry, PersonaRegistry):
        cls.reset_instance()


class TestKeyHelpers:
    def test_make_key(self):
        assert make_key("n") == "n"
        assert make_key("n", "v") == "v:n"
        assert make_key("n", "") == "n"

    def test_split_qualified_name(self):
        assert split_qualified_name("v:n") == ("v", "n")
        assert split_qualified_name("a:b:c") == ("a", "b:c")
        assert split_qualified_name("n") == (None, "n")
        # An explicit vertical wins; the name is kept verbatim.
        assert split_qualified_name("x:n", "v") == ("v", "x:n")


class TestSingletonPerSubclass:
    def test_each_subclass_has_its_own_instance(self):
        toy = _ToyRegistry.get_instance()
        other = _OtherToyRegistry.get_instance()
        assert toy is _ToyRegistry.get_instance()
        assert other is not toy
        assert type(other) is _OtherToyRegistry
        assert ChainRegistry.get_instance() is not PersonaRegistry.get_instance()
        assert isinstance(ChainRegistry.get_instance(), ChainRegistry)
        assert isinstance(PersonaRegistry.get_instance(), PersonaRegistry)

    def test_subclass_does_not_inherit_parent_instance(self):
        parent = _ToyRegistry.get_instance()
        assert _OtherToyRegistry._instance is None
        assert _OtherToyRegistry.get_instance() is not parent

    def test_fresh_subclass_never_sees_parent_instance(self):
        """A subclass defined after the parent singleton exists gets its own."""
        parent = _ToyRegistry.get_instance()

        class _LateSubclass(_ToyRegistry):
            pass

        assert "_instance" in vars(_LateSubclass)
        late = _LateSubclass.get_instance()
        assert late is not parent
        assert type(late) is _LateSubclass
        assert _ToyRegistry.get_instance() is parent

    def test_reset_only_affects_that_subclass(self):
        toy = _ToyRegistry.get_instance()
        chain = ChainRegistry.get_instance()
        _ToyRegistry.reset_instance()
        assert _ToyRegistry.get_instance() is not toy
        assert ChainRegistry.get_instance() is chain

    def test_subclasses_have_distinct_class_locks(self):
        assert _ToyRegistry._class_lock is not _OtherToyRegistry._class_lock
        assert ChainRegistry._class_lock is not PersonaRegistry._class_lock

    def test_concurrent_get_instance_returns_one_object(self):
        seen: List[Any] = []
        barrier = threading.Barrier(8)

        def grab():
            barrier.wait()
            seen.append(_ToyRegistry.get_instance())

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(x) for x in seen}) == 1


class TestBatchUpdate:
    def test_reentrant_nesting_and_depth(self):
        reg = _ToyRegistry()
        with reg.batch_update():
            with reg.batch_update():
                reg.register("a", _Item("red"))
            reg.register("b", _Item("blue"))
        assert _lock_released(reg)
        assert reg.has("a") and reg.has("b")

    def test_depth_restored_after_exception(self):
        reg = _ToyRegistry()
        with pytest.raises(ValueError):
            with reg.batch_update():
                raise ValueError("x")
        assert _lock_released(reg)

    def test_other_threads_block_until_batch_exits(self):
        reg = _ToyRegistry()
        done = threading.Event()

        def writer():
            reg.register("late", _Item("red"))
            done.set()

        with reg.batch_update():
            t = threading.Thread(target=writer)
            t.start()
            assert not done.wait(0.1)
        t.join(2)
        assert done.is_set() and reg.has("late")


class TestIndexes:
    def test_add_and_lookup(self):
        reg = _ToyRegistry()
        item = _Item("red", tags=["x", "y"])
        reg.register("a", item, vertical="v")
        assert reg._index_keys("colour", "red") == {"v:a"}
        assert reg._index_keys("tag", "y") == {"v:a"}
        assert reg._index_keys("tag", "missing") == set()

    def test_replace_removes_stale_terms_and_empty_entries(self):
        reg = _ToyRegistry()
        reg.register("a", _Item("red", tags=["old"]))
        reg.register("a", _Item("blue", tags=["new"]), replace=True)
        assert reg._indexes == {"colour": {"blue": {"a"}}, "tag": {"new": {"a"}}}

    def test_removal_keeps_terms_shared_with_other_keys(self):
        reg = _ToyRegistry()
        reg.register("a", _Item("red", tags=["t"]))
        reg.register("b", _Item("red"))
        assert reg.unregister("a") is True
        assert reg._indexes == {"colour": {"red": {"b"}}, "tag": {}}

    def test_duplicate_without_replace_is_skipped_and_warns(self, caplog):
        reg = _ToyRegistry()
        first = _Item("red")
        reg.register("a", first)
        with caplog.at_level(logging.WARNING, logger=_test_logger.name):
            assert reg.register("a", _Item("blue")) is False
        assert reg.get("a") is first
        assert reg._index_keys("colour", "blue") == set()
        assert "Toy 'a' already registered, skipping" in caplog.text

    def test_match_index_any_and_all(self):
        reg = _ToyRegistry()
        reg.register("a", _Item("r", tags=["x", "y"]))
        reg.register("b", _Item("r", tags=["y"]))
        assert reg._match_index("tag", ["x", "y"], match_all=False) == {"a", "b"}
        assert reg._match_index("tag", ["x", "y"], match_all=True) == {"a"}
        assert reg._match_index("tag", ["x", "zz"], match_all=True) == set()
        assert reg._match_index("tag", ["zz", "x"], match_all=True) == set()

    def test_match_all_does_not_mutate_index(self):
        reg = _ToyRegistry()
        reg.register("a", _Item("r", tags=["x"]))
        reg.register("b", _Item("r", tags=["x", "y"]))
        reg._match_index("tag", ["x", "y"], match_all=True)
        assert reg._index_keys("tag", "x") == {"a", "b"}


class TestLookup:
    def test_has_is_exact_get_falls_back(self):
        reg = _ToyRegistry()
        bare = _Item("r")
        reg.register("n", bare)
        assert reg.has("n") and not reg.has("n", vertical="v")
        assert reg.get("n", vertical="v") is bare
        scoped = _Item("b")
        reg.register("n", scoped, vertical="v")
        assert reg.get("n", vertical="v") is scoped
        assert reg.get("missing", vertical="v") is None

    def test_has_sees_factories(self):
        reg = _ToyRegistry()
        assert reg._put_factory("v:f", lambda: _Item("r"), replace=False)
        assert reg.has("f", vertical="v")
        assert reg.get("f", vertical="v") is None

    def test_resolve_factory_fallbacks(self):
        reg = _ToyRegistry()
        bare = lambda: _Item("r")  # noqa: E731
        reg._put_factory("f", bare, replace=False)
        assert reg._resolve_factory("f", "v") == ("v:f", "v", bare)
        assert reg._resolve_factory("v:f", None) == ("v:f", "v", bare)
        assert reg._resolve_factory("g", None) == ("g", None, None)

    def test_put_factory_duplicate_level(self, caplog):
        reg = _ToyRegistry()
        reg.register("a", _Item("r"))
        with caplog.at_level(logging.DEBUG, logger=_test_logger.name):
            assert not reg._put_factory("a", lambda: None, replace=False, duplicate_level=10)
        records = [r for r in caplog.records if "already registered" in r.message]
        assert [r.levelname for r in records] == ["DEBUG"]
        assert reg._put_factory("a", lambda: None, replace=True)

    def test_list_keys_and_factories(self):
        reg = _ToyRegistry()
        reg.register("a", _Item("r"), vertical="v")
        reg.register("b", _Item("r"))
        reg._put_factory("v:f", lambda: None, replace=False)
        assert reg._list_keys(reg._items, None) == ["v:a", "b"]
        assert reg._list_keys(reg._items, "v") == ["v:a"]
        assert reg.list_factories() == ["v:f"]
        assert reg.list_factories(vertical="w") == []

    def test_projection_helpers_skip_unknown_keys(self):
        reg = _ToyRegistry()
        item = _Item("r")
        reg.register("a", item)
        assert reg._items_dict(["a", "ghost"]) == {"a": item}
        assert reg._items_list(["ghost", "a"]) == [item]


class TestUnregisterAndClear:
    def test_unregister_item_and_factory(self):
        reg = _ToyRegistry()
        reg.register("a", _Item("r"))
        reg._put_factory("a", lambda: None, replace=True)
        assert reg.unregister("a") is True
        assert not reg.has("a")
        assert reg.forgotten == ["a"]
        assert reg.unregister("a") is False

    def test_unregister_value_that_is_falsy(self):
        reg: NamespacedRegistry[Any] = ChainRegistry()
        reg._put("z", None, replace=False, index_source=_FakeMeta())
        assert reg.unregister("z") is True
        assert not reg.has("z")

    def test_clear_resets_everything_and_keeps_index_views(self):
        reg = _ToyRegistry()
        tag_view = reg._indexes["tag"]
        reg.register("a", _Item("r", tags=["t"]))
        reg._put_factory("f", lambda: None, replace=False)
        reg.clear()
        assert reg._items == {} and reg._factories == {}
        assert reg._indexes == {"colour": {}, "tag": {}}
        assert reg._indexes["tag"] is tag_view
        assert reg.extra_cleared == 1


@dataclass
class _FakeMeta:
    tags: List[str] = field(default_factory=list)
    vertical: Optional[str] = None


class TestRegisterFromVertical:
    def test_counts_and_namespaces(self, caplog):
        reg = _ToyRegistry()
        with caplog.at_level(logging.INFO, logger=_test_logger.name):
            count = reg.register_from_vertical("v", {"a": _Item("r"), "b": _Item("g")})
        assert count == 2
        assert reg._list_keys(reg._items, "v") == ["v:a", "v:b"]
        assert "Registered 2 toys from vertical 'v'" in caplog.text

    def test_failures_are_logged_and_not_counted(self, caplog):
        reg = _ToyRegistry()

        def register(name: str, item: Any) -> None:
            if name == "bad":
                raise ValueError("nope")
            reg.register(name, item, vertical="v")

        with caplog.at_level(logging.WARNING, logger=_test_logger.name):
            count = reg._register_from_vertical(
                "v", {"ok": _Item("r"), "bad": _Item("r")}, register
            )
        assert count == 1
        assert "Failed to register v:bad: nope" in caplog.text


class TestThreadSafetySmoke:
    def test_concurrent_writers_and_readers(self):
        reg = _ToyRegistry()
        errors: List[BaseException] = []

        def writer(offset: int) -> None:
            try:
                for i in range(200):
                    reg.register(f"k{offset}_{i}", _Item("r", tags=[f"t{i % 5}"]), replace=True)
                    if i % 3 == 0:
                        reg.unregister(f"k{offset}_{i}")
            except BaseException as e:  # pragma: no cover - surfaced below
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(200):
                    with reg._lock:
                        reg._match_index("tag", ["t1", "t2"], match_all=False)
                    reg.get("k0_1")
                    reg.list_factories()
            except BaseException as e:  # pragma: no cover - surfaced below
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # 3 writers x 200 keys, every third removed
        assert len(reg._items) == 3 * (200 - 67)
        indexed = set().union(*reg._indexes["tag"].values())
        assert indexed == set(reg._items)


class TestHasFactory:
    def test_exact_key_only(self):
        registry = ChainRegistry()
        registry.register_factory("f", lambda: "made", vertical="v")
        registry.register("d", "direct", vertical="v")

        assert registry.has_factory("f", vertical="v")
        assert not registry.has_factory("f")  # no bare-name fallback
        assert not registry.has_factory("d", vertical="v")  # direct entry, not a factory
        assert not registry.has_factory("missing", vertical="v")


def _lock_released(registry) -> bool:
    """True when another thread can take the registry lock (no batch left open)."""
    acquired = []

    def probe() -> None:
        ok = registry._lock.acquire(timeout=1)
        acquired.append(ok)
        if ok:
            registry._lock.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return acquired[0]
