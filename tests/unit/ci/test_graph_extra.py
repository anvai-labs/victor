"""victor-ai[graph] must install every grammar the graph dependency check demands.

A clean `pip install "victor-ai[graph]"` failed require_graph_dependencies()
for 8 languages: the extra and the check's language table had drifted apart.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

from victor.core.graph_rag.language_parsers import LANGUAGE_MODULES

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _extras() -> dict[str, list[Requirement]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return {
        name: [Requirement(raw) for raw in raws]
        for name, raws in data["project"]["optional-dependencies"].items()
    }


def _module(requirement: Requirement) -> str:
    return requirement.name.lower().replace("-", "_")


def test_graph_extra_installs_every_grammar_the_check_demands():
    graph = {_module(r) for r in _extras()["graph"]}
    demanded = {module for module, _ in LANGUAGE_MODULES.values()}
    assert demanded <= graph, sorted(demanded - graph)


def test_graph_grammars_use_the_lang_extra_specifiers():
    extras = _extras()
    lang = {
        r.name.lower(): str(r.specifier)
        for name, requirements in extras.items()
        if name.startswith("lang-") and name != "lang-all"
        for r in requirements
    }
    for requirement in extras["graph"]:
        name = requirement.name.lower()
        if name.startswith("tree-sitter-") and name in lang:
            assert str(requirement.specifier) == lang[name], name


def test_graph_extra_installs_the_enhanced_analysis_plugin():
    assert "victor-coding" in {r.name.lower() for r in _extras()["graph"]}


def test_graph_grammars_carry_no_platform_markers():
    """A marker silently drops a language on some platform; every grammar here
    ships wheels for Linux x86-64/aarch64, macOS and Windows."""
    for requirement in _extras()["graph"]:
        if requirement.name.lower().startswith("tree-sitter-"):
            assert requirement.marker is None, str(requirement)


def test_doctor_floor_matches_the_extra():
    from packaging.version import Version

    from victor.ui.commands.doctor import MIN_VICTOR_CODING

    coding = next(r for r in _extras()["graph"] if r.name.lower() == "victor-coding")
    floors = [s.version for s in coding.specifier if s.operator in (">=", "==", "~=")]
    assert floors and Version(MIN_VICTOR_CODING) == Version(floors[0])
