"""Tier-2.5 scope binding on the tree-sitter path (ADR-032).

Family goldens (JS, Rust, Go) pin the full contract; the verified long tail
(Java, C, Bash) pins per-language facts; every language NOT yet in
``REFERENCE_LANGUAGES`` asserts empty output — membership is honest capability
(a wrong reference is worse than an absent one), and these emptiness pins are
the ratchet that makes adding a language a deliberate, verified act.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_language_pack")

from victor_codegraph import CodeRelationType, parse  # noqa: E402
from victor_codegraph.scope_binding import REFERENCE_LANGUAGES  # noqa: E402
from victor_codegraph.treesitter_parser import GrammarUnavailable  # noqa: E402

JS = """\
import { format } from "./util.js";

class Widget {
  constructor() {
    this.items = [];
  }
  prepare(spec) {
    const copy = spec;
    return format(copy);
  }
}

function build() {
  const w = new Widget();
  const spec = Widget;
  return [w, spec];
}
"""

RUST = """\
use crate::engine::Task;

pub struct Runner {
    task: Task,
}

impl Runner {
    pub fn execute(&self) -> bool {
        let local = Task;
        self.task.is_ready()
    }
}

pub fn standalone() -> Runner {
    Runner
}
"""

GO = """\
package audit

type Collector struct {
    sink map[string]int
}

func (c *Collector) Ingest(name string) {
    c.sink[name] += 1
}

func NewCollector() *Collector {
    c := &Collector{}
    return c
}

func Analyze(collector *Collector) int {
    total := 0
    return total
}
"""

JAVA = """\
public class Service {
    public int handle(Request req) {
        Helper tool = new Helper();
        return tool.size(req);
    }
}
"""

C = """\
#include <stdlib.h>

typedef struct Node { int v; } Node;

int compute(Node *n) {
    Node *other = n;
    return other->v;
}
"""

BASH = """\
#!/usr/bin/env bash
build_target() {
  local artifact="app.bin"
  cp "$artifact" dist/
}

build_target
"""

# Languages not yet in REFERENCE_LANGUAGES, with the pending reason recorded in
# scope_binding's table comment.
PENDING = {
    "cpp": ("model.cpp", "int run() { return 1; }\n"),
    "csharp": ("Handler.cs", "public class Handler { public int Process() { return 1; } }\n"),
    "php": ("Registry.php", "<?php\nfunction add($i) { return $i; }\n"),
    "swift": ("Store.swift", "struct Store { func count() -> Int { return 1 } }\n"),
    "scala": ("Pipe.scala", "object Pipe { def apply(i: Int) = i + 1 }\n"),
    "lua": ("init.lua", "local M = {}\nfunction M.setup(opts) end\nreturn M\n"),
    "kotlin": ("Repo.kt", "class Repo { fun load(q: String) = q.length }\n"),
    "ruby": ("worker.rb", "class Worker\n  def run(task)\n    task\n  end\nend\n"),
    "sql": ("q.sql", "SELECT a FROM t WHERE a > 1;\n"),
}


def _parse(content: str, language: str, file: str):
    try:
        return parse(content, language=language, file_path=file, emit_references=True)
    except Exception as exc:  # GrammarUnavailable and parse errors both skip
        if isinstance(exc, GrammarUnavailable) or "grammar" in str(exc).lower():
            pytest.skip(f"grammar unavailable: {exc}")
        raise


def _refs(parsed):
    return [r for r in parsed.relations if r.relation_type == CodeRelationType.REFERENCES]


def _names(parsed, relations):
    by_id = {s.id: s.simple_name for s in parsed.symbols}
    return [
        by_id.get(r.to_symbol_id, r.to_symbol_id) if r.target_ref is None else r.target_ref.name
        for r in relations
    ]


def test_javascript_family():
    parsed = _parse(JS, "javascript", "widget.js")
    prepare = next(s for s in parsed.symbols if s.simple_name == "prepare")
    build = next(s for s in parsed.symbols if s.simple_name == "build")

    prepare_refs = [r for r in _refs(parsed) if r.from_symbol_id == prepare.id]
    # `spec` is a parameter (shadowed), `format(...)` is a callee (CALLS).
    assert prepare_refs == []

    build_refs = [r for r in _refs(parsed) if r.from_symbol_id == build.id]
    # `new Widget()` is a constructor call (CALLS); the bare `const spec =
    # Widget` value use resolves in-file to the class.
    assert _names(parsed, build_refs) == ["Widget"]
    assert build_refs[0].confidence == 1.0 and build_refs[0].target_ref is None


def test_rust_family():
    parsed = _parse(RUST, "rust", "runner.rs")
    execute = next(s for s in parsed.symbols if s.simple_name == "execute")
    standalone = next(s for s in parsed.symbols if s.simple_name == "standalone")
    runner_struct = next(s for s in parsed.symbols if s.simple_name == "Runner")

    execute_refs = [r for r in _refs(parsed) if r.from_symbol_id == execute.id]
    # `Task` is a use-path import used as a value -> retained for cross-file
    # resolution; `self.task.is_ready()` members/callee are not references.
    assert _names(parsed, execute_refs) == ["Task"]
    assert execute_refs[0].confidence == 0.5 and execute_refs[0].target_ref is not None

    standalone_refs = [r for r in _refs(parsed) if r.from_symbol_id == standalone.id]
    assert _names(parsed, standalone_refs) == ["Runner"]  # return-type use resolves

    struct_refs = [r for r in _refs(parsed) if r.from_symbol_id == runner_struct.id]
    assert _names(parsed, struct_refs) == ["Task"]  # field type annotation retained


def test_go_family():
    parsed = _parse(GO, "go", "audit.go")
    analyze = next(s for s in parsed.symbols if s.simple_name == "Analyze")
    ingest = next(s for s in parsed.symbols if s.simple_name == "Ingest")
    collector = next(s for s in parsed.symbols if s.simple_name == "Collector")

    # Receiver and parameter type uses resolve in-file to the struct.
    for owner in (analyze, ingest):
        refs = [r for r in _refs(parsed) if r.from_symbol_id == owner.id]
        assert [r.to_symbol_id for r in refs] == [collector.id]
        assert refs[0].confidence == 1.0

    # Receiver var, locals and map members never leak as references.
    all_names = set(_names(parsed, _refs(parsed)))
    assert not ({"c", "name", "total", "collector", "sink"} & all_names)


def test_java_verified():
    parsed = _parse(JAVA, "java", "Service.java")
    handle = next(s for s in parsed.symbols if s.simple_name == "handle")
    refs = [r for r in _refs(parsed) if r.from_symbol_id == handle.id]
    # Parameter TYPE use retained for cross-file resolution; the parameter
    # name and the local `tool` are shadowed; `new Helper()` is CALLS.
    assert _names(parsed, refs) == ["Request"]
    assert refs[0].confidence == 0.5 and refs[0].target_ref is not None


def test_c_verified():
    parsed = _parse(C, "c", "model.c")
    compute = next(s for s in parsed.symbols if s.simple_name == "compute")
    node = next(s for s in parsed.symbols if s.simple_name == "Node")
    refs = [r for r in _refs(parsed) if r.from_symbol_id == compute.id]
    # Parameter type use resolves in-file to the typedef'd struct; the local
    # `other` and the member `v` are shadowed/excluded.
    assert [r.to_symbol_id for r in refs] == [node.id]
    assert refs[0].confidence == 1.0


def test_bash_verified():
    parsed = _parse(BASH, "bash", "deploy.sh")
    build_target = next(s for s in parsed.symbols if s.simple_name == "build_target")
    refs = [r for r in _refs(parsed) if r.from_symbol_id == build_target.id]
    # External command retained unresolved; the local `artifact` is shadowed;
    # the top-level `build_target` call is module-level (no owner, skipped).
    assert _names(parsed, refs) == ["cp"]
    assert refs[0].confidence == 0.5


@pytest.mark.parametrize("language", sorted(PENDING), ids=sorted(PENDING))
def test_pending_languages_emit_nothing(language):
    file, source = PENDING[language]
    parsed = _parse(source, language, file)
    assert language not in REFERENCE_LANGUAGES
    assert _refs(parsed) == [], "pending languages must not emit references"


def test_default_off_no_references_any_language():
    for language, source in (
        ("javascript", JS),
        ("rust", RUST),
        ("go", GO),
        ("java", JAVA),
        ("c", C),
        ("bash", BASH),
    ):
        parsed = parse(source, language=language, file_path="x")
        assert _refs(parsed) == [], language
