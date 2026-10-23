"""Graph settings surface (``victor/config/search_settings.py``).

Mirrors the config module per the ci-fast changed-file selection contract and
pins the ADR-032 tier-2.5 flag: default OFF, and the GraphRagConfig cascade
honoring the settings field. Also pins the tier-3 keys (ADR-032 rev 1.8):
``precise_index_timeout_seconds``/``precise_index_tool`` must exist on the
settings group — without them the from_sources() yaml merge drops the keys
before they can reach the pipeline — and the typed ``graph_setting``
accessor must preserve non-bool types (``graph_flag`` coerces with bool()).
"""

from __future__ import annotations

from types import SimpleNamespace

from victor.config.search_settings import GraphSettings

_GRAPH_ENV_VARS = (
    "VICTOR_SEARCH__GRAPH__ENABLE_PRECISE_INDEX",
    "VICTOR_SEARCH__GRAPH__PRECISE_INDEX_TIMEOUT_SECONDS",
    "VICTOR_SEARCH__GRAPH__PRECISE_INDEX_TOOL",
    "VICTOR_SEARCH__GRAPH__ENABLE_SCOPE_BINDING",
)


def _nested_settings(**graph_overrides):
    graph = SimpleNamespace(
        enable_ccg=True,
        enable_graph_rag=True,
        **graph_overrides,
    )
    return SimpleNamespace(search=SimpleNamespace(graph=graph))


def test_scope_binding_graduated_default_on():
    assert GraphSettings().enable_scope_binding is True


def test_precise_index_settings_defaults():
    """Tier 3 stays opt-in with provider-default timeout semantics (0)."""
    settings = GraphSettings()
    assert settings.enable_precise_index is False
    assert settings.precise_index_timeout_seconds == 0
    assert settings.precise_index_tool == "auto"


def test_precise_index_settings_flow_from_yaml_sources(tmp_path, monkeypatch):
    """settings.yaml is the documented way to enable tier 3 — the keys must
    survive the Settings.from_sources() merge with no env vars set (the
    CLI/init/watcher construction sites read through from_sources())."""
    from victor.config.settings import Settings

    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "settings.yaml").write_text(
        "search:\n"
        "  graph:\n"
        "    enable_precise_index: true\n"
        "    precise_index_timeout_seconds: 3600\n"
        "    precise_index_tool: scip-python\n"
    )

    settings = Settings.from_sources(config_dir=tmp_path)

    assert settings.search.graph.enable_precise_index is True
    assert settings.search.graph.precise_index_timeout_seconds == 3600
    assert settings.search.graph.precise_index_tool == "scip-python"


def test_graph_setting_preserves_declared_default_type():
    """graph_setting must return the declared default's type for non-bool
    keys — the bool-only graph_flag collapses numeric defaults (0 → False)."""
    from victor.config.settings import Settings, graph_flag, graph_setting

    settings = Settings()
    timeout = graph_setting(settings, "precise_index_timeout_seconds", 0)
    assert timeout == 0
    assert isinstance(timeout, int) and timeout is not False

    # Missing key → declared default, type intact.
    bare = _nested_settings()
    assert graph_setting(bare, "precise_index_timeout_seconds", 3600) == 3600

    # None on the group falls back to the default (tri-state overrides).
    none_valued = _nested_settings(precise_index_timeout_seconds=None)
    assert graph_setting(none_valued, "precise_index_timeout_seconds", 0) == 0

    # The bool-only accessor keeps its documented coercion contract.
    assert graph_flag(_nested_settings(enable_precise_index=True), "enable_precise_index", False)


def test_graph_rag_config_reads_scope_binding_from_settings(tmp_path):
    from victor.core.graph_rag.config import GraphRagConfig

    config = GraphRagConfig.from_settings(
        root_path=tmp_path,
        settings=_nested_settings(enable_scope_binding=True),
    )
    assert config.index.enable_scope_binding is True

    default = GraphRagConfig.from_settings(
        root_path=tmp_path,
        settings=_nested_settings(),
    )
    assert default.index.enable_scope_binding is True  # graduated default

    opt_out = GraphRagConfig.from_settings(
        root_path=tmp_path,
        settings=_nested_settings(enable_scope_binding=False),
    )
    assert opt_out.index.enable_scope_binding is False  # explicit opt-out honored


def test_graph_rag_config_wires_precise_index_settings(tmp_path):
    """from_settings must read the nested search.graph group and carry the
    tier-3 timeout/tool through — flat getattr silently missed every key."""
    from victor.core.graph_rag.config import GraphRagConfig

    config = GraphRagConfig.from_settings(
        root_path=tmp_path,
        settings=_nested_settings(
            enable_precise_index=True,
            precise_index_timeout_seconds=3600,
            precise_index_tool="scip-python",
        ),
    )
    assert config.index.enable_precise_index is True
    assert config.index.precise_index_timeout_seconds == 3600
    assert config.index.precise_index_tool == "scip-python"

    default = GraphRagConfig.from_settings(root_path=tmp_path, settings=_nested_settings())
    assert default.index.enable_precise_index is False
    assert default.index.precise_index_timeout_seconds == 0
    assert default.index.precise_index_tool == "auto"
