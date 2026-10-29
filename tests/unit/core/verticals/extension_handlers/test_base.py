"""Shared dict-extension loader used by the chains and personas handlers."""

from __future__ import annotations

import sys
import types
from typing import Any, Callable, ClassVar, Dict, List

import pytest

from victor.core.verticals.extension_handlers.base import _load_dict_extension


def _make_ctx(candidates: List[str], name: str = "my-vertical") -> type:
    class FakeLoader:
        cache: ClassVar[Dict[str, Any]] = {}
        requested_suffixes: ClassVar[List[str]] = []

        @classmethod
        def _find_available_candidates(cls, suffix: str) -> List[str]:
            cls.requested_suffixes.append(suffix)
            return list(candidates)

        @classmethod
        def _get_cached_extension(cls, key: str, factory: Callable[[], Any]) -> Any:
            if key not in cls.cache:
                cls.cache[key] = factory()
            return cls.cache[key]

    FakeLoader.name = name  # type: ignore[attr-defined]
    return FakeLoader


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    counter = iter(range(1000))

    def make(**attrs: Any) -> str:
        mod_name = f"_victor_test_dict_ext_{next(counter)}"
        module = types.ModuleType(mod_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, mod_name, module)
        return mod_name

    return make


def _load(ctx: type) -> Dict[str, Any]:
    return _load_dict_extension(
        ctx,  # type: ignore[arg-type]
        suffix="sfx",
        constant_suffix="THINGS",
        factory_name="get_things",
        cache_key="things",
    )


def test_no_candidates_returns_empty_and_queries_suffix() -> None:
    ctx = _make_ctx([])
    assert _load(ctx) == {}
    assert ctx.requested_suffixes == ["sfx"]


def test_factory_wins_over_constant_and_is_cached(fake_module: Callable[..., str]) -> None:
    mod = fake_module(get_things=lambda: {"f": 1}, MY_VERTICAL_THINGS={"c": 1})
    ctx = _make_ctx([mod])
    assert _load(ctx) == {"f": 1}
    assert ctx.cache["things"] == {"f": 1}


def test_factory_returning_none_yields_empty_dict(fake_module: Callable[..., str]) -> None:
    ctx = _make_ctx([fake_module(get_things=lambda: None)])
    assert _load(ctx) == {}


def test_constant_name_uses_upper_snake_vertical_name(fake_module: Callable[..., str]) -> None:
    constant = {"c": 2}
    ctx = _make_ctx([fake_module(MY_VERTICAL_THINGS=constant)])
    result = _load(ctx)
    assert result == constant and result is not constant


def test_non_dict_constant_and_missing_attrs_fall_through(
    fake_module: Callable[..., str],
) -> None:
    first = fake_module(MY_VERTICAL_THINGS=["not", "a", "dict"], get_things="not callable")
    second = fake_module(MY_VERTICAL_THINGS={"second": True})
    assert _load(_make_ctx([first, second])) == {"second": True}


def test_import_error_is_skipped_when_a_later_candidate_provides(
    fake_module: Callable[..., str],
) -> None:
    good = fake_module(get_things=lambda: {"ok": 1})
    assert _load(_make_ctx(["_victor_missing_module_xyz", good])) == {"ok": 1}


def test_last_import_error_reraised_when_nothing_provides(
    fake_module: Callable[..., str],
) -> None:
    empty = fake_module()
    with pytest.raises(ImportError, match="_victor_missing_module_b"):
        _load(_make_ctx(["_victor_missing_module_a", empty, "_victor_missing_module_b"]))


def test_empty_when_candidates_import_but_provide_nothing(
    fake_module: Callable[..., str],
) -> None:
    assert _load(_make_ctx([fake_module()])) == {}
