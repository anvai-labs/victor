"""ChainsHandler resolves through the shared dict loader (personas: test_personas.py)."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from victor.core.verticals.extension_handlers import base as base_module
from victor.core.verticals.extension_handlers.chains import ChainsHandler
from victor.core.verticals.extension_handlers.personas import PersonasHandler


@pytest.mark.parametrize(
    "handler,expected",
    [
        (
            ChainsHandler,
            {
                "suffix": "composed_chains",
                "constant_suffix": "CHAINS",
                "factory_name": "get_composed_chains",
                "cache_key": "composed_chains",
            },
        ),
    ],
)
def test_handler_passes_its_extension_parameters(
    monkeypatch: pytest.MonkeyPatch, handler: Any, expected: Dict[str, str]
) -> None:
    calls: list[tuple[Any, Dict[str, str]]] = []

    def fake_loader(ctx: Any, **kwargs: str) -> Dict[str, Any]:
        calls.append((ctx, kwargs))
        return {"sentinel": True}

    module = __import__(handler.__module__, fromlist=["_load_dict_extension"])
    monkeypatch.setattr(module, "_load_dict_extension", fake_loader)
    ctx = object()
    assert handler.load(ctx) == {"sentinel": True}
    assert calls == [(ctx, expected)]


def test_handlers_keep_extension_types() -> None:
    assert ChainsHandler.extension_type == "chains"
    assert PersonasHandler.extension_type == "personas"


def test_handler_module_uses_base_loader() -> None:
    import victor.core.verticals.extension_handlers.chains as chains_module
    import victor.core.verticals.extension_handlers.personas as personas_module

    assert chains_module._load_dict_extension is base_module._load_dict_extension
    assert personas_module._load_dict_extension is base_module._load_dict_extension
