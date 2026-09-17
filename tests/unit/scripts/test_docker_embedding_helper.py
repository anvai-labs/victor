"""The offline initializer must report persistence failures, including stale files."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from victor.agent.shared_tool_registry import SharedToolRegistry
from victor.core.data_cache import save_data_with_metadata
from victor.tools import semantic_selector
from victor.tools.decorators import tool


@pytest.mark.parametrize("write_failure", [False, True])
def test_helper_requires_current_persisted_snapshot(tmp_path, monkeypatch, write_failure):
    @tool(name="helper_probe")
    def probe() -> str:
        """A harmless tool for deployment validation."""
        return "ok"

    monkeypatch.setattr(
        SharedToolRegistry,
        "get_instance",
        classmethod(lambda cls: SimpleNamespace(get_all_tools_for_registration=lambda: [probe])),
    )
    original = semantic_selector.SemanticToolSelector

    class Selector(original):
        def __init__(self, **kwargs):
            super().__init__(cache_dir=tmp_path, **kwargs)
            assert save_data_with_metadata(
                self.cache_file,
                {
                    "cache_version": self.CACHE_VERSION,
                    "embedding_model": self.embedding_model,
                    "tools_hash": "stale-tool-definitions",
                    "tool_count": 1,
                    "tool_names": ["old_tool"],
                    "embeddings": {"old_tool": np.ones(384)},
                },
                logger=logging.getLogger(__name__),
                label="Stale helper fixture",
            )

        async def _get_embedding(self, text):
            return np.ones(384)

    monkeypatch.setattr(semantic_selector, "SemanticToolSelector", Selector)
    if write_failure:
        monkeypatch.setattr(semantic_selector, "save_data_with_metadata", lambda *a, **kw: False)
    path = Path(__file__).resolve().parents[3] / "docker/scripts/init-embeddings.sh"
    program = path.read_text().split("python3 <<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]
    if write_failure:
        with pytest.raises(RuntimeError, match="Current tool embeddings were not persisted"):
            exec(compile(program, str(path), "exec"), {"__name__": "__main__"})
    else:
        exec(compile(program, str(path), "exec"), {"__name__": "__main__"})
