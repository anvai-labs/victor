from __future__ import annotations

from victor_codegraph import CapabilityTier, ChunkConfig, ParseStatus, parse


def test_syntax_error_reports_fallback_diagnostic():
    parsed = parse("def broken(:\n", file_path="bad.py")
    assert parsed.status == ParseStatus.FALLBACK
    assert parsed.diagnostics
    assert parsed.diagnostics[0].code == "syntax_error"


def test_unknown_language_reports_fallback_instead_of_silent_success():
    parsed = parse("plain text", language="unknown", file_path="x.unknown")
    assert parsed.status == ParseStatus.FALLBACK
    assert parsed.capability_tier == CapabilityTier.FALLBACK
    assert parsed.diagnostics


def test_extract_relations_config_is_effective():
    parsed = parse(
        "def f():\n    return g()\n",
        file_path="m.py",
        config=ChunkConfig(extract_relations=False),
    )
    assert parsed.relations == []


def test_shadowed_python_declarations_remain_explicit_identity_collisions():
    parsed = parse(
        "class C:\n    @property\n    def value(self): return 1\n"
        "    @property\n    def value(self): return 2\n",
        file_path="duplicate.py",
    )
    assert parsed.status == ParseStatus.PARTIAL
    assert [(item.code, item.severity) for item in parsed.diagnostics] == [
        ("identity_collision", "error"),
        ("identity_collision", "error"),
    ]
    assert [(item.line, item.column) for item in parsed.diagnostics] == [(2, 4), (4, 4)]
    definitions = [symbol for symbol in parsed.symbols if symbol.simple_name == "value"]
    assert len(definitions) == 2
    assert definitions[0].id == definitions[1].id
    assert definitions[0].legacy_id != definitions[1].legacy_id
    assert all("duplicate.py::C::value" in item.message for item in parsed.diagnostics)
    assert all(definitions[0].id in item.message for item in parsed.diagnostics)
