"""PersonasHandler resolves through the shared dict loader."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from victor.core.verticals.extension_handlers import personas as personas_module
from victor.core.verticals.extension_handlers.personas import PersonasHandler


def test_personas_handler_passes_its_extension_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, Dict[str, str]]] = []

    def fake_loader(ctx: Any, **kwargs: str) -> Dict[str, Any]:
        calls.append((ctx, kwargs))
        return {"sentinel": True}

    monkeypatch.setattr(personas_module, "_load_dict_extension", fake_loader)
    ctx = object()
    assert PersonasHandler.load(ctx) == {"sentinel": True}  # type: ignore[arg-type]
    assert calls == [
        (
            ctx,
            {
                "suffix": "teams",
                "constant_suffix": "PERSONAS",
                "factory_name": "get_personas",
                "cache_key": "personas",
            },
        )
    ]
