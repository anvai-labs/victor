"""Python parser tests — run fully offline (stdlib ast, no grammar needed)."""

from __future__ import annotations

from victor_codegraph import CodeSymbolType, parse
from victor_codegraph.model import CodeRelationType

SAMPLE = '''\
"""Module doc."""
import os
from typing import Any


def top_level(x: int, y: str = "a") -> bool:
    """A function."""
    return helper(x)


def helper(x: int) -> int:
    return x + 1


class Greeter(Base):
    """A class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return top_level(1)
'''


def test_extracts_functions_classes_methods():
    parsed = parse(SAMPLE, file_path="pkg/mod.py")
    names = {s.simple_name for s in parsed.symbols}
    assert {"top_level", "helper", "Greeter", "__init__", "greet"} <= names


def test_symbol_types_and_constructor():
    parsed = parse(SAMPLE, file_path="pkg/mod.py")
    by_name = {s.simple_name: s for s in parsed.symbols}
    assert by_name["top_level"].symbol_type == CodeSymbolType.FUNCTION
    assert by_name["greet"].symbol_type == CodeSymbolType.METHOD
    assert by_name["__init__"].symbol_type == CodeSymbolType.CONSTRUCTOR
    assert by_name["Greeter"].symbol_type == CodeSymbolType.CLASS


def test_signature_and_docstring_and_params():
    parsed = parse(SAMPLE, file_path="pkg/mod.py")
    fn = next(s for s in parsed.symbols if s.simple_name == "top_level")
    assert fn.signature == "top_level(x: int, y: str) -> bool"
    assert fn.documentation == "A function."
    assert {p["name"] for p in fn.parameters} == {"x", "y"}
    assert fn.return_type == "bool"


def test_imports():
    parsed = parse(SAMPLE, file_path="pkg/mod.py")
    assert any("import os" in i for i in parsed.imports)
    assert any("from typing import Any" in i for i in parsed.imports)


def test_calls_relation_resolved_to_ids():
    parsed = parse(SAMPLE, file_path="pkg/mod.py")
    ids = {s.id for s in parsed.symbols}
    calls = [r for r in parsed.relations if r.relation_type == CodeRelationType.CALLS]
    assert calls, "expected at least one CALLS edge"
    for r in calls:
        assert r.from_symbol_id in ids
        assert r.to_symbol_id in ids  # resolved to a real symbol, not a bare name


def test_unresolved_callee_is_retained():
    """An outgoing call to a symbol not defined in-file is KEPT (bare-name target,
    confidence < 1.0, with a call_site) so cross-file/external references survive for
    CPG blast-radius / impact analysis rather than being silently dropped."""
    parsed = parse("def authenticate():\n    return verify()\n", file_path="svc/auth.py")
    calls = list(parsed.relations)
    assert len(calls) == 1
    r = calls[0]
    assert r.to_symbol_id == "verify"  # bare name — NOT dropped
    assert r.confidence < 1.0
    assert r.call_site is not None
    assert r.call_site.start_line == 2  # the verify() call line (1-based)


def test_recursive_self_call_is_retained():
    """Recursion is graph evidence and remains a self-edge."""
    parsed = parse("def f():\n    return f()\n", file_path="m.py")
    assert len(parsed.relations) == 1
    assert parsed.relations[0].from_symbol_id == parsed.relations[0].to_symbol_id


def test_extends_relation():
    parsed = parse(SAMPLE, file_path="pkg/mod.py")
    cls = next(s for s in parsed.symbols if s.simple_name == "Greeter")
    assert any("extends(Base)" in m for m in cls.modifiers)


def test_deterministic_ids():
    a = parse(SAMPLE, file_path="pkg/mod.py")
    b = parse(SAMPLE, file_path="pkg/mod.py")
    assert [s.id for s in a.symbols] == [s.id for s in b.symbols]


def test_syntax_error_falls_back_to_no_symbols():
    parsed = parse("def broken(:\n", file_path="bad.py")
    assert parsed.symbols == []
