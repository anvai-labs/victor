"""ADR-032 tier 3 providers — producer grammars, registry, project discovery.

Every grammar case below is pinned to a REAL captured dump (scip-python
0.6.6, scip-typescript 0.5.1, rust-analyzer 1.99, scip-clang 0.4.1,
scip-java 0.13.2, scip-ruby 0.5.1, scip-go 0.2.9) — the #1436 lesson: symbol
grammars are producer properties and documentation is not evidence.
"""

from __future__ import annotations

from pathlib import Path

from victor.core.graph_rag.precise_providers import (
    DumpSkipped,
    RustAnalyzerProvider,
    ScipClangProvider,
    ScipGoProvider,
    ScipJavaProvider,
    ScipPythonProvider,
    ScipRubyProvider,
    ScipTypescriptProvider,
    providers_for_files,
)

PY = ScipPythonProvider()
TS = ScipTypescriptProvider()
RS = RustAnalyzerProvider()
CC = ScipClangProvider()
JV = ScipJavaProvider()
RB = ScipRubyProvider()
GO = ScipGoProvider()


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
    # Polyglot extension (rev 1.9): C/C++ headers route to scip-clang (one
    # producer for both languages); JVM/ruby/go get their own producers.
    providers = providers_for_files(["src/a.c", "src/b.cpp", "inc/x.hpp", "src/y.h"])
    assert set(providers) == {"scip-clang"}
    providers = providers_for_files(["App.java", "lib/toy.rb", "cmd/main.go"])
    assert set(providers) == {"scip-java", "scip-ruby", "scip-go"}
    # .scala stays UNROUTED until its grammar is captured against a real
    # dump — routing it to scip-java today would ship an unverified parser.
    assert providers_for_files(["App.scala"]) == {}


def test_clang_grammar_real_shapes():
    # Prefix has EMPTY package/version tokens as literal '.': "cxx . . $ ".
    assert _path(CC, "cxx . . $ helper(d4f767463ce0a6b3).") == "helper"
    # Namespace components drop from free functions AND type chains.
    assert _path(CC, "cxx . . $ outer/inner/nsfree(7864480464b09eea).") == "nsfree"
    assert _path(CC, "cxx . . $ ns/Greeter#") == "Greeter"
    assert _path(CC, "cxx . . $ Point#") == "Point"  # C struct shares the shape
    assert _path(CC, "cxx . . $ outer/inner/Widget#Handle#") == "Widget.Handle"
    assert _path(CC, "cxx . . $ ns/Greeter#greet(49f6e7a06ebc5aa8).") == "Greeter.greet"
    assert _path(CC, "cxx . . $ outer/inner/Widget#Handle#grab(49f6e7a06ebc5aa8).") == (
        "Widget.Handle.grab"
    )
    # Templates are transparent.
    assert _path(CC, "cxx . . $ Box#") == "Box"
    assert _path(CC, "cxx . . $ Box#get(5f357b03ed472e5e).") == "Box.get"
    # The parenthesized suffix is a SIGNATURE hash: helper(int) and
    # overload(int) share one — identity must ride name+scope, so the hash
    # strips and the two collapse to their (distinct) names.
    assert _path(CC, "cxx . . $ overload(d4f767463ce0a6b3).") == "overload"
    assert _path(CC, "cxx . . $ overload(ba8836b59fb42830).") == "overload"
    # Non-joinable descriptors: namespaces, fields, macros, file pseudos.
    assert _path(CC, "cxx . . $ outer/") is None
    assert _path(CC, "cxx . . $ outer/inner/") is None
    assert _path(CC, "cxx . . $ Box#value.") is None
    assert _path(CC, "cxx . . $ Point#x.") is None
    assert _path(CC, "cxx . . $ `src/a.h:2:9`!") is None
    assert _path(CC, "cxx . . $ `<file>/src/a.c`/") is None
    assert _path(CC, "local 0") is None
    assert _path(CC, "cxx . . $ helper().") is None  # hash-less variant: refuse to guess


def test_java_grammar_real_shapes():
    # Leading tokens VARY per project (no fixed prefix); packages drop.
    P = "scip-java maven maven/com.toy/toy 0.1.0 "
    assert _path(JV, P + "com/toy/App#") == "App"
    assert _path(JV, P + "com/toy/App#helper().") == "App.helper"
    assert _path(JV, P + "com/toy/App#main().") == "App.main"
    assert _path(JV, P + "com/toy/Util#twice().") == "Util.twice"
    # Constructors emit the `<init>` literal — mapped like scip-typescript.
    assert _path(JV, P + "com/toy/App#`<init>`().") == "App.constructor"
    # Nested classes chain '#'; packages still drop.
    assert _path(JV, P + "com/toy/Outer#Inner#") == "Outer.Inner"
    assert _path(JV, P + "com/toy/Outer#Inner#deep().") == "Outer.Inner.deep"
    assert _path(JV, P + "com/toy/Outer#Inner#`<init>`().") == "Outer.Inner.constructor"
    # Fields are non-joinable.
    assert _path(JV, P + "com/toy/App#value.") is None
    assert _path(JV, "local 2") is None
    assert _path(JV, "scip-java maven maven/com.toy/toy 0.1.0") is None


def test_ruby_grammar_real_shapes():
    # Leading tokens vary per gem; module/class nesting is KEPT as scope
    # (ruby modules are real lexical nesting — unlike cxx/java namespaces).
    P = "scip-ruby gem toy 0.1.0 "
    assert _path(RB, P + "Toy#") == "Toy"
    assert _path(RB, P + "Toy#Greeter#") == "Toy.Greeter"
    assert _path(RB, P + "Toy#Greeter#greet().") == "Toy.Greeter.greet"
    assert _path(RB, P + "Toy#Greeter#initialize().") == "Toy.Greeter.initialize"
    # Singleton methods ride the metaclass literal — mapped to the class,
    # INCLUDING in middle position for nested modules.
    assert _path(RB, P + "`<Class:Toy>`#build().") == "Toy.build"
    assert _path(RB, P + "Toy#Inner#") == "Toy.Inner"
    assert _path(RB, P + "Toy#`<Class:Inner>`#helper().") == "Toy.Inner.helper"
    # Ivars are fields: non-joinable.
    assert _path(RB, P + "Toy#Greeter#`@name`.") is None
    assert _path(RB, "local 1$3681232888") is None


def test_go_grammar_real_shapes():
    # FOUR leading tokens (version is the literal '.'); the backticked
    # import-path head drops entirely.
    P = "scip-go gomod example.com/toy . "
    assert _path(GO, P + "`example.com/toy/util`/Box#") == "Box"
    assert _path(GO, P + "`example.com/toy/util`/Box#Get().") == "Box.Get"
    # Pointer receivers share the value-receiver shape (no marker).
    assert _path(GO, P + "`example.com/toy/util`/Box#Set().") == "Box.Set"
    assert _path(GO, P + "`example.com/toy/util`/Twice().") == "Twice"
    assert _path(GO, P + "`example.com/toy`/main().") == "main"
    # Interfaces are type nodes; their method SIGNATURES are non-callable
    # (no '()') and stay unjoinable like fields.
    assert _path(GO, P + "`example.com/toy/util`/Speaker#") == "Speaker"
    assert _path(GO, P + "`example.com/toy/util`/Speaker#Speak.") is None
    assert _path(GO, P + "`example.com/toy/util`/Box#Value.") is None
    assert _path(GO, P + "`example.com/toy/util`/") is None
    assert _path(GO, "local 0") is None


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


def test_clang_compile_database_discovery(tmp_path):
    assert CC.compilation_database(tmp_path) is None
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "compile_commands.json").write_text("[]", encoding="utf-8")
    assert CC.compilation_database(tmp_path) == tmp_path / "build" / "compile_commands.json"
    (tmp_path / "compile_commands.json").write_text("[]", encoding="utf-8")
    assert CC.compilation_database(tmp_path) == tmp_path / "compile_commands.json"
    # Bear-style one-level-down layouts.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "compile_commands.json").write_text("[]", encoding="utf-8")
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    (empty / "sub").mkdir()
    (empty / "sub" / "build" / "compile_commands.json").parent.mkdir(parents=True)
    (empty / "sub" / "build" / "compile_commands.json").write_text("[]", encoding="utf-8")
    assert CC.compilation_database(empty) == empty / "sub" / "build" / "compile_commands.json"


def test_go_module_discovery(tmp_path):
    assert GO.module_root(tmp_path) is None
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "go.mod").write_text("module x\n", encoding="utf-8")
    assert GO.module_root(tmp_path) == tmp_path / "server"
    (tmp_path / "go.mod").write_text("module y\n", encoding="utf-8")
    assert GO.module_root(tmp_path) == tmp_path


def test_clang_dump_argv_runs_from_project_root(tmp_path, monkeypatch):
    # scip-clang filters documents to project_root = CWD: running from the
    # artifact dir silently yields an EMPTY index — cwd must be the repo
    # root (caught by real-pipeline validation, not argv mocks).
    (tmp_path / "compile_commands.json").write_text("[]", encoding="utf-8")
    captured = _capture_dump(monkeypatch, dump_result=Path("/x/index.scip"))
    output = tmp_path / ".victor" / "precise" / "out.scip"
    assert CC.dump("/usr/bin/scip-clang", tmp_path, output, timeout_seconds=42) is not None
    assert captured["command"] == [
        "/usr/bin/scip-clang",
        "--compdb-path",
        str(tmp_path / "compile_commands.json"),
        "--index-output-path",
        str(output),
        "--no-progress-report",
    ]
    assert captured["cwd"] == tmp_path
    assert captured["timeout"] == 42
    assert captured["node_heap"] is False


def test_clang_dump_without_database_skips(tmp_path, monkeypatch):
    captured = _capture_dump(monkeypatch)
    result = CC.dump("/usr/bin/scip-clang", tmp_path, Path("/x.scip"), timeout_seconds=1)
    assert isinstance(result, DumpSkipped) and result.reason == "no-compile-database"
    assert captured == {}  # never reached the transport


def test_java_dump_argv_and_build_marker_gate(tmp_path, monkeypatch):
    output = tmp_path / "precise" / "out.scip"
    captured = _capture_dump(monkeypatch, dump_result=output)
    # No JVM build file → a typed marker skip (NOT dump-failed) before the
    # transport.
    result = JV.dump("/usr/bin/scip-java", tmp_path, output, timeout_seconds=42)
    assert isinstance(result, DumpSkipped) and result.reason == "no-jvm-build"
    assert captured == {}
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    assert JV.dump("/usr/bin/scip-java", tmp_path, output, timeout_seconds=42) is not None
    assert captured["command"] == ["/usr/bin/scip-java", "index", f"--output={output}"]
    assert captured["cwd"] == tmp_path
    assert captured["node_heap"] is False


def test_ruby_dump_argv_bundle_exec_and_fixed_name_move(tmp_path, monkeypatch):
    (tmp_path / "Gemfile").write_text('source "https://rubygems.org"\n', encoding="utf-8")
    output = tmp_path / ".victor" / "precise" / "out.scip"
    output.parent.mkdir(parents=True, exist_ok=True)
    captured = {}

    def fake_run_dump(command, *, cwd, output_path, timeout_seconds, node_heap):
        captured.update(command=command, cwd=cwd, output=output_path, timeout=timeout_seconds)
        # Simulate scip-ruby's fixed-name contract: index.scip lands in CWD.
        output_path.write_bytes(b"dump")
        return output_path

    import victor.core.graph_rag.precise_providers as mod

    monkeypatch.setattr(mod, "_run_dump", fake_run_dump)
    assert RB.dump("/usr/bin/bundle", tmp_path, output, timeout_seconds=42) is not None
    assert captured["command"] == ["/usr/bin/bundle", "exec", "scip-ruby", "."]
    assert captured["cwd"] == tmp_path
    assert captured["output"] == tmp_path / "index.scip"
    assert output.read_bytes() == b"dump"  # moved to the artifact path
    assert not (tmp_path / "index.scip").exists()


def test_ruby_dump_without_gemfile_skips(tmp_path, monkeypatch):
    captured = _capture_dump(monkeypatch)
    result = RB.dump("/usr/bin/bundle", tmp_path, Path("/x.scip"), timeout_seconds=1)
    assert isinstance(result, DumpSkipped) and result.reason == "no-gemfile"
    assert captured == {}


def test_marker_gates_return_typed_skips_not_failures(tmp_path, monkeypatch):
    """The honest taxonomy's third state: project marker absent ≠ toolchain
    or dump failure. Rust/TS/Go gates join the clang/java/ruby ones."""
    captured = _capture_dump(monkeypatch)

    def assert_skip(provider, tool, reason):
        result = provider.dump(tool, tmp_path, Path("/x.scip"), timeout_seconds=1)
        assert isinstance(result, DumpSkipped) and result.reason == reason, provider.name

    assert_skip(RS, "/usr/bin/rust-analyzer", "no-cargo-toml")
    assert_skip(TS, "/usr/bin/scip-typescript", "no-tsconfig")
    assert_skip(GO, "/usr/bin/scip-go", "no-go-module")
    assert captured == {}  # no transport call for any of them


def test_go_dump_argv_fixed_name_move_from_module_root(tmp_path, monkeypatch):
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    output = tmp_path / ".victor" / "precise" / "out.scip"
    output.parent.mkdir(parents=True, exist_ok=True)
    captured = {}

    def fake_run_dump(command, *, cwd, output_path, timeout_seconds, node_heap):
        captured.update(command=command, cwd=cwd)
        output_path.write_bytes(b"dump")
        return output_path

    import victor.core.graph_rag.precise_providers as mod

    monkeypatch.setattr(mod, "_run_dump", fake_run_dump)
    assert GO.dump("/usr/bin/scip-go", tmp_path, output, timeout_seconds=42) is not None
    assert captured["command"] == ["/usr/bin/scip-go", "index"]
    assert captured["cwd"] == tmp_path
    assert output.read_bytes() == b"dump"
    assert not (tmp_path / "index.scip").exists()


def test_detection_for_polyglot_providers(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert CC.detect() == "/usr/bin/scip-clang"
    assert JV.detect() == "/usr/bin/scip-java"
    assert GO.detect() == "/usr/bin/scip-go"
    # scip-ruby needs BOTH bundler and the gem (the wrapper refuses direct
    # invocation); the hint override names the bundler to run through.
    assert RB.detect() == "/usr/bin/bundle"
    which_calls = []

    def which(name):
        which_calls.append(name)
        return None if name == "scip-ruby" else f"/usr/bin/{name}"

    monkeypatch.setattr("shutil.which", which)
    assert RB.detect() is None  # gem absent → toolchain-absent, not a lie
    explicit = tmp_path / "bundler"
    explicit.write_text("", encoding="utf-8")
    assert RB.detect(str(explicit)) == str(explicit)


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


def test_clang_join_end_to_end_hash_and_cross_file(tmp_path):
    """Captured scip-clang shapes: cross-file refs carry the identical
    signature-hash string; namespaces drop; fields/file-pseudos never pool."""
    import pytest

    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2
    from victor.core.graph_rag.precise_index import join_occurrences

    class N:
        def __init__(self, node_id, name, file, type_, parent_id=None):
            self.node_id, self.name, self.file, self.type = node_id, name, file, type_
            self.parent_id = parent_id

    index = scip_pb2.Index()
    doc_a = index.documents.add()
    doc_a.relative_path = "src/a.c"
    doc_b = index.documents.add()
    doc_b.relative_path = "src/b.c"
    rows_a = [
        ("cxx . . $ `<file>/src/a.c`/", 0, 1),
        ("cxx . . $ helper(d4f767463ce0a6b3).", 2, 1),
    ]
    rows_b = [
        ("cxx . . $ `<file>/src/b.c`/", 0, 1),
        ("cxx . . $ main(b126dc7c1de90089).", 2, 1),
        ("cxx . . $ helper(d4f767463ce0a6b3).", 4, 0),  # cross-file ref
    ]
    for doc, rows in ((doc_a, rows_a), (doc_b, rows_b)):
        for symbol, line, roles in rows:
            occ = doc.occurrences.add()
            occ.symbol = symbol
            occ.symbol_roles = roles
            occ.range.extend([line, 0])

    nodes = {
        "src/a.c": [N("helper", "helper", "src/a.c", "function")],
        "src/b.c": [N("main", "main", "src/b.c", "function")],
    }
    harvest = join_occurrences(index, nodes, symbol_parts=CC.symbol_parts)
    assert harvest.edges == [
        (
            "main",
            "helper",
            "REFERENCES",
            {"provenance": "compiler", "confidence": 1.0, "use": "load"},
        )
    ]
    assert harvest.unjoined == 0  # file pseudo-symbols never pool


def test_java_join_end_to_end_and_honest_constructor_gap(tmp_path):
    """Captured scip-java shapes: methods join through the class-parented
    scope; the `<init>` constructor has NO codegraph node (the java
    extractor emits none) and must stay honestly unjoined, never guessed."""
    import pytest

    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2
    from victor.core.graph_rag.precise_index import join_occurrences

    class N:
        def __init__(self, node_id, name, file, type_, parent_id=None):
            self.node_id, self.name, self.file, self.type = node_id, name, file, type_
            self.parent_id = parent_id

    P = "scip-java maven maven/com.toy/toy 0.1.0 "
    index = scip_pb2.Index()
    doc = index.documents.add()
    doc.relative_path = "src/main/java/com/toy/App.java"
    rows = [
        (P + "com/toy/App#", 0, 1),
        (P + "com/toy/App#`<init>`().", 2, 1),
        (P + "com/toy/App#helper().", 5, 1),
        (P + "com/toy/App#main().", 8, 1),
        (P + "com/toy/App#helper().", 9, 0),  # ref inside main
        (P + "com/toy/App#value.", 4, 1),  # field def: never pools
    ]
    for symbol, line, roles in rows:
        occ = doc.occurrences.add()
        occ.symbol = symbol
        occ.symbol_roles = roles
        occ.range.extend([line, 0])

    # Measured codegraph java shape: class + methods, NO constructor node.
    nodes = [
        N("app", "App", "src/main/java/com/toy/App.java", "class"),
        N("helper", "helper", "src/main/java/com/toy/App.java", "method", "app"),
        N("main", "main", "src/main/java/com/toy/App.java", "method", "app"),
    ]
    harvest = join_occurrences(
        index, {"src/main/java/com/toy/App.java": nodes}, symbol_parts=JV.symbol_parts
    )
    assert harvest.edges == [
        (
            "main",
            "helper",
            "REFERENCES",
            {"provenance": "compiler", "confidence": 1.0, "use": "load"},
        )
    ]
    assert harvest.definitions_unjoined == 1  # <init>: honest, not guessed


def test_ruby_join_refuses_false_flat_joins(tmp_path):
    """Captured scip-ruby shapes vs the MEASURED codegraph ruby extractor
    (flat functions, file-named module — no class/singleton nodes): the
    producer-correct paths must NOT be flattened to force joins."""
    import pytest

    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2
    from victor.core.graph_rag.precise_index import join_occurrences

    class N:
        def __init__(self, node_id, name, file, type_, parent_id=None):
            self.node_id, self.name, self.file, self.type = node_id, name, file, type_
            self.parent_id = parent_id

    P = "scip-ruby gem toy 0.1.0 "
    index = scip_pb2.Index()
    doc = index.documents.add()
    doc.relative_path = "lib/toy.rb"
    rows = [
        (P + "Toy#Greeter#greet().", 3, 1),
        (P + "`<Class:Toy>`#build().", 9, 1),
        (P + "Toy#Greeter#greet().", 12, 0),  # ref at the call site
    ]
    for symbol, line, roles in rows:
        occ = doc.occurrences.add()
        occ.symbol = symbol
        occ.symbol_roles = roles
        occ.range.extend([line, 0])

    # Measured extractor output: flat greet function, no Greeter/build nodes.
    nodes = [N("greet", "greet", "lib/toy.rb", "function")]
    harvest = join_occurrences(index, {"lib/toy.rb": nodes}, symbol_parts=RB.symbol_parts)
    assert harvest.edges == []  # no false flat join: Toy.Greeter.greet ≠ greet
    assert harvest.resolved == 0
    # The unjoined decomposition rides the join's ref-dedup rules (refs
    # whose owning def never joined collapse into it) — the contract here is
    # the honest-zero: nothing resolved, nothing guessed.
    assert harvest.unjoined >= 1


def test_go_join_end_to_end_perfect_alignment(tmp_path):
    """Captured scip-go shapes: receiver methods join through the struct
    scope; import-path heads drop; the measured toy corpus joined 100%."""
    import pytest

    pytest.importorskip("google.protobuf")
    from victor.core.graph_rag._scip import scip_pb2
    from victor.core.graph_rag.precise_index import join_occurrences

    class N:
        def __init__(self, node_id, name, file, type_, parent_id=None):
            self.node_id, self.name, self.file, self.type = node_id, name, file, type_
            self.parent_id = parent_id

    P = "scip-go gomod example.com/toy . "
    index = scip_pb2.Index()
    doc_main = index.documents.add()
    doc_main.relative_path = "main.go"
    doc_util = index.documents.add()
    doc_util.relative_path = "util/util.go"
    rows_util = [
        (P + "`example.com/toy/util`/", 0, 1),
        (P + "`example.com/toy/util`/Box#", 3, 1),
        (P + "`example.com/toy/util`/Box#Get().", 6, 1),
        (P + "`example.com/toy/util`/Box#Set().", 10, 1),
        (P + "`example.com/toy/util`/Twice().", 13, 1),
    ]
    rows_main = [
        (P + "`example.com/toy`/", 0, 1),
        (P + "`example.com/toy`/main().", 6, 1),
        (P + "`example.com/toy/util`/Twice().", 8, 0),  # cross-file ref
        (P + "`example.com/toy/util`/Box#Get().", 10, 0),
    ]
    for doc, rows in ((doc_main, rows_main), (doc_util, rows_util)):
        for symbol, line, roles in rows:
            occ = doc.occurrences.add()
            occ.symbol = symbol
            occ.symbol_roles = roles
            occ.range.extend([line, 0])

    nodes = {
        "main.go": [N("main", "main", "main.go", "function")],
        "util/util.go": [
            N("box", "Box", "util/util.go", "class"),
            N("get", "Get", "util/util.go", "method", "box"),
            N("set", "Set", "util/util.go", "method", "box"),
            N("twice", "Twice", "util/util.go", "function"),
        ],
    }
    harvest = join_occurrences(index, nodes, symbol_parts=GO.symbol_parts)
    pairs = {e[:2] for e in harvest.edges}
    assert pairs == {("main", "twice"), ("main", "get")}
    assert harvest.unjoined == 0


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
    # No tsconfig anywhere -> typed marker skip before any subprocess.
    result = TS.dump("/usr/bin/scip-typescript", tmp_path, tmp_path / "o.scip", timeout_seconds=60)
    assert isinstance(result, DumpSkipped) and result.reason == "no-tsconfig"
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
    result = RS.dump("/usr/bin/rust-analyzer", tmp_path, tmp_path / "o.scip", timeout_seconds=60)
    assert isinstance(result, DumpSkipped) and result.reason == "no-cargo-toml"
    assert "command" not in captured  # no Cargo.toml -> skip
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    out_dir = tmp_path / "artifact"
    # rust-analyzer writes index.scip to the process CWD regardless of the
    # requested path: the transport validates the FIXED name and the
    # provider MOVES it to the caller's path (caught by a real pipeline
    # validation — argv-level tests alone could not see this).
    import victor.core.graph_rag.precise_providers as mod

    def fake_run_dump(command, *, cwd, output_path, timeout_seconds, node_heap):
        captured.update(
            command=command,
            cwd=cwd,
            output=output_path,
            timeout=timeout_seconds,
            node_heap=node_heap,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"scip")
        return output_path

    monkeypatch.setattr(mod, "_run_dump", fake_run_dump)
    final = RS.dump("/usr/bin/rust-analyzer", tmp_path, out_dir / "rust.scip", timeout_seconds=60)
    assert final == out_dir / "rust.scip"
    assert captured["command"] == ["/usr/bin/rust-analyzer", "scip", str(tmp_path / "rust")]
    assert captured["cwd"] == out_dir
    assert captured["output"] == out_dir / "index.scip"  # the FIXED name
    assert captured["node_heap"] is False  # native producer: no Node hint
    assert final.exists() and final.read_bytes() == b"scip"
    assert not (out_dir / "index.scip").exists()  # moved, not copied


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
