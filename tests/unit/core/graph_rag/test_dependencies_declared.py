"""The graph dependency check demands exactly what victor-ai[graph] declares here."""

import importlib.metadata

import pytest

import victor.core.graph_rag.dependencies as deps


def _requires(monkeypatch, requirements):
    def fake(name):
        assert name == "victor-ai"
        return requirements

    monkeypatch.setattr(importlib.metadata, "requires", fake)


def test_declared_grammars_follow_extra_and_platform_markers(monkeypatch):
    _requires(
        monkeypatch,
        [
            'tree-sitter-java>=0.23; extra == "graph"',
            'tree-sitter-c-sharp>=0.23; extra == "graph"',
            'tree-sitter-lua>=0.2; (sys_platform != "linux" or platform_machine != "aarch64") '
            'and extra == "graph"',
            'tree-sitter-ruby>=0.23; extra == "lang-ruby"',
            'tree-sitter-language-pack>=1.21; extra == "graph"',
            "tree-sitter>=0.26",
        ],
    )
    declared = deps.declared_graph_grammars()
    assert {"tree_sitter_java", "tree_sitter_c_sharp"} <= declared
    assert "tree_sitter_ruby" not in declared  # another extra
    assert "tree_sitter_language_pack" not in declared
    assert "tree_sitter" not in declared  # unconditional base dependency, not a grammar


def test_no_distribution_metadata_means_every_grammar_is_required(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "requires", missing)
    assert deps.declared_graph_grammars() is None


def test_an_undeclared_grammar_is_not_demanded(monkeypatch):
    """A grammar with no wheel on this platform is neither installed nor required."""
    import victor.core.graph_rag.indexing as indexing

    monkeypatch.setattr(
        indexing,
        "_TREE_SITTER_LANGUAGE_MODULES",
        {"nowheel": ("tree_sitter_definitely_absent", "language")},
    )
    monkeypatch.setattr(deps, "declared_graph_grammars", lambda: frozenset())
    pytest.importorskip("victor_codegraph")
    pytest.importorskip("tree_sitter")
    deps.require_graph_dependencies()  # must not raise

    monkeypatch.setattr(deps, "declared_graph_grammars", lambda: None)
    with pytest.raises(RuntimeError, match="tree_sitter_definitely_absent"):
        deps.require_graph_dependencies()
