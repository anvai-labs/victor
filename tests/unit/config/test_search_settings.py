"""Graph settings surface (``victor/config/search_settings.py``).

Mirrors the config module per the ci-fast changed-file selection contract and
pins the ADR-032 tier-2.5 flag: default OFF, and the GraphRagConfig cascade
honoring the settings field.
"""

from __future__ import annotations

from types import SimpleNamespace

from victor.config.search_settings import GraphSettings


def test_scope_binding_graduated_default_on():
    assert GraphSettings().enable_scope_binding is True


def test_graph_rag_config_reads_scope_binding_from_settings(tmp_path):
    from victor.core.graph_rag.config import GraphRagConfig

    config = GraphRagConfig.from_settings(
        root_path=tmp_path,
        settings=SimpleNamespace(enable_ccg=True, enable_graph_rag=True, enable_scope_binding=True),
    )
    assert config.index.enable_scope_binding is True

    default = GraphRagConfig.from_settings(
        root_path=tmp_path,
        settings=SimpleNamespace(enable_ccg=True, enable_graph_rag=True),
    )
    assert default.index.enable_scope_binding is True  # graduated default

    opt_out = GraphRagConfig.from_settings(
        root_path=tmp_path,
        settings=SimpleNamespace(
            enable_ccg=True, enable_graph_rag=True, enable_scope_binding=False
        ),
    )
    assert opt_out.index.enable_scope_binding is False  # explicit opt-out honored
