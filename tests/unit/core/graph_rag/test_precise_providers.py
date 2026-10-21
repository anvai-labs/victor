"""ADR-032 tier 3 providers — producer grammars, registry, project discovery.

Every grammar case below is pinned to a REAL captured dump (scip-python
0.6.6, scip-typescript 0.5.1, rust-analyzer 1.99) — the #1436 lesson: symbol
grammars are producer properties and documentation is not evidence.
"""

from __future__ import annotations

from pathlib import Path

from victor.core.graph_rag.precise_providers import (
    RustAnalyzerProvider,
    ScipPythonProvider,
    ScipTypescriptProvider,
    providers_for_files,
)

PY = ScipPythonProvider()
TS = ScipTypescriptProvider()
RS = RustAnalyzerProvider()


def _path(provider, symbol):
    parts = provider.symbol_parts(symbol)
    return parts["path"] if parts else None


def test_python_grammar_real_shapes():
    assert _path(PY, "scip-python python fixture 0.1.0 lib/helper().") == "helper"
    assert _path(PY, "scip-python python fixture 0.1.0 lib/Worker#run().") == "Worker.run"
    assert _path(PY, "scip-python python fixture 0.1.0 lib/Outer#Inner#m().") == "Outer.Inner.m"
    assert (
        _path(PY, "scip-python python victor-harvest 0.1.0 `victor.core.mod`/thing().") == "thing"
    )
    assert _path(PY, "scip-python python fixture 0.1.0 lib/Worker#") == "Worker"
    assert _path(PY, "scip-python python fixture 0.1.0 lib/__init__:") is None
    assert _path(PY, "scip-python python fixture 0.1.0 lib/helper().(name)") is None
    assert _path(PY, "scip-python python fixture 0.1.0 lib/Worker#enabled.") is None
    assert _path(PY, "local 0") is None
    assert _path(PY, "rust crate::x") is None


def test_typescript_grammar_real_shapes():
    # The TS head is a backticked FILE path, not a dotted module.
    assert _path(TS, "scip-typescript npm . . src/`greeter.ts`/Greeter#greet().") == "Greeter.greet"
    assert (
        _path(TS, "scip-typescript npm . . src/`greeter.ts`/Greeter#`<constructor>`().")
        == "Greeter.constructor"
    )
    assert _path(TS, "scip-typescript npm . . src/`greeter.ts`/helper().") == "helper"
    assert _path(TS, "scip-typescript npm . . src/`greeter.ts`/Greeter#") == "Greeter"
    assert _path(TS, "scip-typescript npm . . src/`greeter.ts`/Greeter#name.") is None
    assert _path(TS, "scip-typescript npm . . src/`greeter.ts`/") is None
    assert _path(TS, "local 1") is None


def test_rust_grammar_real_shapes():
    # Top-level items carry NO module prefix (unlike python).
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 helper().") == "helper"
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 extra/extra_fn().") == "extra_fn"
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 Greeter#") == "Greeter"
    # impl#[Type]member() — the impl'd type IS the codegraph scope (methods
    # parent directly to the struct node).
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 impl#[Greeter]new().") == "Greeter.new"
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 impl#[Greeter]greet().") == "Greeter.greet"
    # Modules, fields, macros: non-joinable descriptors.
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 crate/") is None
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 extra/") is None
    assert _path(RS, "rust-analyzer cargo fx 0.1.0 Greeter#name.") is None
    assert _path(RS, "rust-analyzer cargo alloc https://x macros/format!") is None
    assert _path(RS, "local 3") is None
    # External (alloc/core) symbols parse to parts — the JOIN filters them
    # as out-of-corpus (defined nowhere in our index); parsing is honest,
    # filtering is the join's job.
    assert _path(RS, "rust-analyzer cargo core https://x ops/arith/Add#add().") == "Add.add"


def test_registry_routes_extensions_to_providers():
    providers = providers_for_files(["src/a.py", "src/b.ts"])
    assert set(providers) == {"scip-python", "scip-typescript"}
    providers = providers_for_files(["src/a.rs", "src/b.tsx", "src/c.js"])
    assert set(providers) == {"rust-analyzer", "scip-typescript"}
    assert providers_for_files(["README.md", "img.svg"]) == {}
    # Monorepo reality: all three at once.
    providers = providers_for_files(["victor/x.py", "vscode-victor/src/y.ts", "rust/z.rs"])
    assert set(providers) == {"scip-python", "scip-typescript", "rust-analyzer"}


def test_typescript_project_discovery(tmp_path):
    # Root tsconfig indexes the root; a first-level subdir's is included
    # (victor's vscode-victor layout); nothing recursive.
    (tmp_path / "vscode-victor").mkdir()
    (tmp_path / "vscode-victor" / "tsconfig.json").write_text("{}", encoding="utf-8")
    assert TS.tsconfig_projects(tmp_path) == [tmp_path / "vscode-victor"]
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    assert TS.tsconfig_projects(tmp_path) == [tmp_path, tmp_path / "vscode-victor"]
    assert TS.tsconfig_projects(tmp_path / "empty") == []


def test_rust_crate_discovery(tmp_path):
    assert RS.crate_root(tmp_path) is None
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    assert RS.crate_root(tmp_path) == tmp_path / "rust"
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert RS.crate_root(tmp_path) == tmp_path


def test_typescript_join_end_to_end_with_kinds(tmp_path):
    """Captured TS dump shapes through the join: kinds ARE populated by this
    producer, so the kind filter is live; the constructor joins by its
    codegraph-literal name."""
    import pytest

    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2
    from victor.core.graph_rag.precise_index import join_occurrences

    def N(node_id, name, file, type_, parent=None):  # GraphNode-shaped
        return SimpleNamespaceShim(node_id, name, file, type_, parent)

    class SimpleNamespaceShim:
        def __init__(self, node_id, name, file, type_, parent_id):
            self.node_id, self.name, self.file, self.type = node_id, name, file, type_
            self.parent_id = parent_id

    index = scip_pb2.Index()
    doc = index.documents.add()
    doc.relative_path = "src/greeter.ts"
    table = {
        "scip-typescript npm . . src/`greeter.ts`/Greeter#": 7,  # Class
        "scip-typescript npm . . src/`greeter.ts`/Greeter#`<constructor>`().": 9,
        "scip-typescript npm . . src/`greeter.ts`/Greeter#greet().": 26,  # Method
        "scip-typescript npm . . src/`greeter.ts`/helper().": 17,  # Function
    }
    for symbol, kind in table.items():
        info = doc.symbols.add()
        info.symbol = symbol
        info.kind = kind
    rows = [
        ("scip-typescript npm . . src/`greeter.ts`/Greeter#", 1, 1),
        ("scip-typescript npm . . src/`greeter.ts`/Greeter#`<constructor>`().", 2, 1),
        ("scip-typescript npm . . src/`greeter.ts`/Greeter#greet().", 3, 1),
        ("scip-typescript npm . . src/`greeter.ts`/helper().", 5, 1),
        ("scip-typescript npm . . src/`greeter.ts`/helper().", 3, 0),  # ref in greet
    ]
    for symbol, line, roles in rows:
        occ = doc.occurrences.add()
        occ.symbol = symbol
        occ.symbol_roles = roles
        occ.range.extend([line, 0])

    nodes = [
        N("greeter", "Greeter", "src/greeter.ts", "class"),
        N("ctor", "constructor", "src/greeter.ts", "constructor", parent="greeter"),
        N("greet", "greet", "src/greeter.ts", "method", parent="greeter"),
        N("helper", "helper", "src/greeter.ts", "function"),
    ]
    harvest = join_occurrences(index, {"src/greeter.ts": nodes}, symbol_parts=TS.symbol_parts)
    assert harvest.edges == [
        (
            "greet",
            "helper",
            "REFERENCES",
            {"provenance": "compiler", "confidence": 1.0, "use": "load"},
        )
    ]


def test_rust_join_end_to_end_impl_scope(tmp_path):
    """Captured rust dump shapes: impl#[T]member() joins through the struct-
    parented scope chain; externals are out-of-corpus and uncounted."""
    import pytest

    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2
    from victor.core.graph_rag.precise_index import join_occurrences

    class N:
        def __init__(self, node_id, name, file, type_, parent_id=None):
            self.node_id, self.name, self.file, self.type = node_id, name, file, type_
            self.parent_id = parent_id

    index = scip_pb2.Index()
    doc = index.documents.add()
    doc.relative_path = "src/lib.rs"
    rows = [
        ("rust-analyzer cargo fx 0.1.0 crate/", 0, 1),
        ("rust-analyzer cargo fx 0.1.0 Greeter#", 2, 1),
        ("rust-analyzer cargo fx 0.1.0 impl#[Greeter]new().", 5, 1),
        ("rust-analyzer cargo fx 0.1.0 impl#[Greeter]greet().", 8, 1),
        ("rust-analyzer cargo fx 0.1.0 helper().", 13, 1),
        ("rust-analyzer cargo fx 0.1.0 caller().", 17, 1),
        ("rust-analyzer cargo alloc https://x string/String#", 1, 0),  # external
        ("rust-analyzer cargo fx 0.1.0 helper().", 9, 0),  # ref inside greet
        ("rust-analyzer cargo fx 0.1.0 impl#[Greeter]new().", 18, 0),  # in caller
        ("rust-analyzer cargo fx 0.1.0 impl#[Greeter]greet().", 19, 0),
        ("rust-analyzer cargo fx 0.1.0 helper().", 19, 0),
    ]
    for symbol, line, roles in rows:
        occ = doc.occurrences.add()
        occ.symbol = symbol
        occ.symbol_roles = roles
        occ.range.extend([line, 0])

    nodes = [
        N("lib", "lib", "src/lib.rs", "module"),
        N("greeter", "Greeter", "src/lib.rs", "struct", parent_id="lib"),
        N("new", "new", "src/lib.rs", "constructor", parent_id="greeter"),
        N("greet", "greet", "src/lib.rs", "method", parent_id="greeter"),
        N("helper", "helper", "src/lib.rs", "function", parent_id="lib"),
        N("caller", "caller", "src/lib.rs", "function", parent_id="lib"),
    ]
    harvest = join_occurrences(index, {"src/lib.rs": nodes}, symbol_parts=RS.symbol_parts)
    pairs = {e[:2] for e in harvest.edges}
    assert pairs == {
        ("greet", "helper"),
        ("caller", "new"),
        ("caller", "greet"),
        ("caller", "helper"),
    }
    assert harvest.unjoined == 0  # the external String ref is out-of-corpus


def _capture_dump(monkeypatch, dump_result=None):
    """Capture the argv each provider builds; fake the shared transport."""
    import victor.core.graph_rag.precise_providers as mod

    captured = {}

    def fake_run_dump(command, *, cwd, output_path, timeout_seconds, node_heap):
        captured.update(
            command=command,
            cwd=cwd,
            output=output_path,
            timeout=timeout_seconds,
            node_heap=node_heap,
        )
        return dump_result

    monkeypatch.setattr(mod, "_run_dump", fake_run_dump)
    return captured


def test_detection_paths_and_hints(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert PY.detect() == "/usr/bin/scip-python"
    assert TS.detect("auto") == "/usr/bin/scip-typescript"
    assert RS.detect("auto") == "/usr/bin/rust-analyzer"
    # An explicit existing path wins over PATH for every provider.
    explicit = tmp_path / "custom-tool"
    explicit.write_text("", encoding="utf-8")
    assert PY.detect(str(explicit)) == str(explicit)
    assert TS.detect(str(explicit)) == str(explicit)
    assert RS.detect(str(explicit)) == str(explicit)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert PY.detect() is None and TS.detect() is None and RS.detect() is None


def test_python_dump_argv(tmp_path, monkeypatch):
    captured = _capture_dump(monkeypatch, dump_result=tmp_path / "out.scip")
    out = PY.dump("/usr/bin/scip-python", tmp_path, tmp_path / "out.scip", timeout_seconds=60)
    assert out == tmp_path / "out.scip"
    assert captured["command"] == [
        "/usr/bin/scip-python",
        "index",
        "--project-name",
        "victor-harvest",
        "--output",
        str(tmp_path / "out.scip"),
    ]
    assert captured["cwd"] == tmp_path
    assert captured["node_heap"] is True  # Node producer: heap hint applies


def test_typescript_dump_argv_and_no_project_skip(tmp_path, monkeypatch):
    captured = _capture_dump(monkeypatch)
    # No tsconfig anywhere -> explicit skip before any subprocess.
    assert (
        TS.dump("/usr/bin/scip-typescript", tmp_path, tmp_path / "o.scip", timeout_seconds=60)
        is None
    )
    assert "command" not in captured
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "tsconfig.json").write_text("{}", encoding="utf-8")
    TS.dump("/usr/bin/scip-typescript", tmp_path, tmp_path / "o.scip", timeout_seconds=60)
    assert captured["command"] == [
        "/usr/bin/scip-typescript",
        "index",
        "--cwd",
        str(tmp_path),
        "--output",
        str(tmp_path / "o.scip"),
        str(tmp_path),
        str(tmp_path / "ui"),
    ]


def test_rust_dump_argv_runs_from_artifact_dir(tmp_path, monkeypatch):
    captured = _capture_dump(monkeypatch)
    assert (
        RS.dump("/usr/bin/rust-analyzer", tmp_path, tmp_path / "o.scip", timeout_seconds=60) is None
    )
    assert "command" not in captured  # no Cargo.toml -> skip
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    out_dir = tmp_path / "artifact"
    RS.dump("/usr/bin/rust-analyzer", tmp_path, out_dir / "index.scip", timeout_seconds=60)
    # rust-analyzer writes index.scip to the process CWD: cwd IS the
    # artifact directory, and the path arg is the discovered crate root.
    assert captured["command"] == ["/usr/bin/rust-analyzer", "scip", str(tmp_path / "rust")]
    assert captured["cwd"] == out_dir
    assert captured["node_heap"] is False  # native producer: no Node hint


def test_run_dump_failure_taxonomy(tmp_path, monkeypatch):
    import subprocess as subprocess_mod

    import victor.core.graph_rag.precise_providers as mod

    out = tmp_path / "index.scip"

    class Fake:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess_mod, "run", lambda *a, **k: Fake())
    assert (
        mod._run_dump(["t"], cwd=tmp_path, output_path=out, timeout_seconds=5, node_heap=False)
        is None
    )

    def boom_timeout(*a, **k):
        raise subprocess_mod.TimeoutExpired(cmd="t", timeout=5)

    monkeypatch.setattr(subprocess_mod, "run", boom_timeout)
    assert (
        mod._run_dump(["t"], cwd=tmp_path, output_path=out, timeout_seconds=5, node_heap=False)
        is None
    )

    def boom_os(*a, **k):
        raise OSError("spawn failed")

    monkeypatch.setattr(subprocess_mod, "run", boom_os)
    assert (
        mod._run_dump(["t"], cwd=tmp_path, output_path=out, timeout_seconds=5, node_heap=False)
        is None
    )

    class Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    captured_env = {}

    def ok_run(cmd, *, env, **k):
        captured_env.update(env)
        out.write_bytes(b"x")
        return Ok()

    monkeypatch.setattr(subprocess_mod, "run", ok_run)
    assert (
        mod._run_dump(["t"], cwd=tmp_path, output_path=out, timeout_seconds=5, node_heap=True)
        == out
    )
    assert captured_env["NODE_OPTIONS"] == mod.NODE_HEAP_OPTIONS


def test_run_dump_preserves_operator_node_options(tmp_path, monkeypatch):
    import subprocess as subprocess_mod

    import victor.core.graph_rag.precise_providers as mod

    out = tmp_path / "index.scip"
    captured_env = {}

    class Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    def ok_run(cmd, *, env, **k):
        captured_env.update(env)
        out.write_bytes(b"x")
        return Ok()

    monkeypatch.setattr(subprocess_mod, "run", ok_run)
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=512")
    mod._run_dump(["t"], cwd=tmp_path, output_path=out, timeout_seconds=5, node_heap=True)
    assert captured_env["NODE_OPTIONS"] == "--max-old-space-size=512"
