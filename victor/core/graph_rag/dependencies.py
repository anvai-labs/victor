"""Explicit installation boundary for the graph CLI's structural capabilities."""

from importlib import import_module
from typing import Optional


def declared_graph_grammars() -> Optional[frozenset[str]]:
    """Grammar modules the installed ``victor-ai[graph]`` declares for this platform.

    Read from the distribution's own Requires-Dist, so the check below
    demands exactly what ``pip install "victor-ai[graph]"`` installs here —
    the extra and the check cannot drift apart (they did: a clean install
    failed the check for 8 languages). Platform markers count: a grammar
    without a wheel on this platform is neither installed nor required.
    None when no distribution metadata exists (an uninstalled source tree);
    the check then requires every known grammar.
    """
    from importlib.metadata import PackageNotFoundError, requires

    from packaging.requirements import InvalidRequirement, Requirement

    try:
        declared = requires("victor-ai")
    except PackageNotFoundError:
        return None
    modules: set[str] = set()
    for raw in declared or []:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        name = requirement.name.lower()
        if not name.startswith("tree-sitter-") or name == "tree-sitter-language-pack":
            continue
        if requirement.marker is None or not requirement.marker.evaluate({"extra": "graph"}):
            continue
        modules.add(name.replace("-", "_"))
    return frozenset(modules) or None


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

    declared = declared_graph_grammars()
    for language, (module, factory) in _TREE_SITTER_LANGUAGE_MODULES.items():
        if declared is not None and module not in declared:
            continue  # not installed by victor-ai[graph] on this platform
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
