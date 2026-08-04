from __future__ import annotations

import pytest
from pathlib import Path

from victor.core.codegraph_adapter import (
    CodeImport,
    parse_code,
    project_python_imports,
)


def test_parse_code_preserves_canonical_symbols_and_structured_imports() -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    content = (
        "import os, sys\n"
        "from pathlib import Path, PurePath as PP\n\n"
        "class Worker:\n"
        "    def run(self, value: int) -> int:\n"
        "        return value\n"
    )

    parsed = parse_code(content, file_path="pkg/worker.py", language="python")

    assert parsed is not None
    expected = codegraph.parse(content, file_path="pkg/worker.py", language="python")
    assert [symbol.id for symbol in parsed.symbols] == [symbol.id for symbol in expected.symbols]
    assert project_python_imports(parsed.imports) == [
        CodeImport(module="os"),
        CodeImport(module="sys"),
        CodeImport(module="pathlib", names=("Path", "PurePath"), is_from_import=True),
    ]


def test_parse_code_is_a_soft_boundary(monkeypatch) -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    monkeypatch.setattr(
        codegraph,
        "parse",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    assert parse_code("def f(): pass", file_path="f.py", language="python") is None


def test_parse_code_normalizes_identity_against_repository_root(tmp_path: Path) -> None:
    pytest.importorskip("victor_codegraph")
    source = tmp_path / "pkg" / "worker.py"
    source.parent.mkdir()

    parsed = parse_code(
        "def work():\n    pass\n",
        file_path=str(source),
        language="python",
        repo_root=tmp_path,
    )

    assert parsed is not None
    assert parsed.file_path == "pkg/worker.py"
    assert {symbol.location.file_path for symbol in parsed.symbols} == {"pkg/worker.py"}
