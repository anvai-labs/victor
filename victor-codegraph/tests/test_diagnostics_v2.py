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
