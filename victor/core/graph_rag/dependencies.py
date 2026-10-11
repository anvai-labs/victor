"""Explicit installation boundary for the graph CLI's structural capabilities."""

from importlib import import_module


def require_graph_dependencies() -> None:
    """Require usable repository parsing and CCG grammars before index mutation.

    Installation remains an explicit package-manager operation: a graph command
    must never install Python packages or silently reduce its fidelity. The
    language pack may initialize its own grammar cache during this check.
    """
    from victor.core.graph_rag.indexing import _TREE_SITTER_LANGUAGE_MODULES

    problems: list[str] = []
    try:
        codegraph = import_module("victor_codegraph")
    except ImportError:
        codegraph = None
        problems.append("victor_codegraph")
    try:
        tree_sitter = import_module("tree_sitter")
    except ImportError:
        tree_sitter = None
        problems.append("tree_sitter")

    for language, (module, factory) in _TREE_SITTER_LANGUAGE_MODULES.items():
        try:
            grammar = getattr(import_module(module), factory)()
            if tree_sitter is not None:
                tree_sitter.Parser(tree_sitter.Language(grammar))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            problems.append(module)
        if codegraph is not None:
            parsed = codegraph.parse("", language=language, file_path=f"probe.{language}")
            if parsed.status.value in {"fallback", "error"}:
                problems.append(f"victor_codegraph grammar: {language}")

    if problems:
        raise RuntimeError(
            "Graph dependencies missing or unusable: "
            + ", ".join(problems)
            + '. Install into the Victor environment: pip install "victor-ai[graph]" '
            + "Existing index was not changed."
        )
