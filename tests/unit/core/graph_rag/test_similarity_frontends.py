"""Frontend capability, lexical binding and partial recovery contracts."""

import pytest

from victor.core.graph_rag.structural_clones import report_to_dict, scan


def _pair(root, extension, left, right, **options):
    (root / f"left.{extension}").write_text(left, encoding="utf-8")
    (root / f"right.{extension}").write_text(right, encoding="utf-8")
    return scan(root, min_nodes=5, near_metric="subtree", near_threshold=1, **options)


@pytest.mark.parametrize(
    "extension,left,right",
    [
        (
            "rs",
            "fn f(xs:Vec<i32>)->i32 { xs.iter().map(|item| item+1).sum() }",
            "fn g(rows:Vec<i32>)->i32 { rows.iter().map(|entry| entry+1).sum() }",
        ),
        (
            "rs",
            "fn f(x:i32)->i32 { match x { 0=>1, n if n>1=>n*3, _=>2 } }",
            "fn g(arg:i32)->i32 { match arg { 0=>1, value if value>1=>value*3, _=>2 } }",
        ),
        (
            "rs",
            "fn f(x:i32)->i32 { let y=x+1; {let y=x+2;consume(y);} y+3 }",
            "fn g(arg:i32)->i32 { let outer=arg+1; {let inner=arg+2;consume(inner);} outer+3 }",
        ),
        (
            "js",
            "function f(x) { const g=(y)=>x-y; return g(x)+2; }",
            "function h(a) { const callback=(b)=>a-b; return callback(a)+2; }",
        ),
        (
            "py",
            "def f(x):\n    callback = lambda y: x-y\n    return callback(x)+2\n",
            "def g(a):\n    invoke = lambda b: a-b\n    return invoke(a)+2\n",
        ),
        (
            "py",
            "def f(rows):\n    result = [item+1 for item in rows if item>2]\n    return len(result)+3\n",
            "def g(values):\n    out = [entry+1 for entry in values if entry>2]\n    return len(out)+3\n",
        ),
    ],
)
def test_scoped_renames_reach_normalized_exact(tmp_path, extension, left, right):
    report = _pair(tmp_path, extension, left, right)
    assert any(f.tier == "exact" for f in report.families), report.stats


@pytest.mark.parametrize(
    "extension,left,right",
    [
        (
            "rs",
            "fn f(x:i32)->i32 { let callback=|y| y+x; callback(x) }",
            "fn f(x:i32)->i32 { let callback=|y| x+x; callback(x) }",
        ),
        (
            "js",
            "function f(x) { const g=(y)=>y-x; return g(x); }",
            "function f(x) { const g=(y)=>x-x; return g(x); }",
        ),
        (
            "py",
            "def f(x):\n    callback = lambda y: y-x\n    return callback(x)\n",
            "def f(x):\n    callback = lambda y: x-x\n    return callback(x)\n",
        ),
        (
            "py",
            "def f(x):\n    return [x+item for item in x]\n",
            "def f(x):\n    return [item+item for item in x]\n",
        ),
    ],
)
def test_capture_and_parameter_roles_are_not_erased(tmp_path, extension, left, right):
    assert _pair(tmp_path, extension, left, right).families == []


@pytest.mark.parametrize(
    "extension,source",
    [
        ("kt", "fun compute(x:Int,y:Int):Int { val value=x+y; return value*3+4 }"),
        ("scala", "object A { def compute(x:Int,y:Int):Int = { val value=x+y; value*3+4 } }"),
        ("cs", "class A { int Compute(int x,int y) { int value=x+y; return value*3+4; } }"),
        ("php", "<?php function compute($x,$y) { $value=$x+$y; return $value*3+4; }"),
        ("swift", "func compute(_ x:Int,_ y:Int)->Int { let value = x + y; return value * 3 + 4 }"),
        ("lua", "function compute(x,y) local value=x+y; return value*3+4 end"),
    ],
)
def test_syntax_frontend_does_not_require_a_scip_provider(tmp_path, extension, source):
    import re

    right = re.sub(
        r"\b(x|y|value)\b", lambda m: {"x": "first", "y": "second", "value": "result"}[m[0]], source
    )
    report = _pair(tmp_path, extension, source, right)
    assert report.coverage_complete, report.stats
    assert report.stats["functions"] == 2
    assert report.stats["scip_dumps"] == []
    assert any(f.tier == "exact" for f in report.families)


def test_valid_rust_unit_survives_an_unrelated_parse_error(tmp_path):
    clean = "fn compute(x:i32)->i32 { let result=x+1; result*3+4 }\n"
    report = _pair(tmp_path, "rs", clean, clean + "\nfn broken( { not valid")
    assert not report.coverage_complete
    assert report.stats["parse_error_files"] == 1
    assert report.stats["recovered_units"] == 1
    (family,) = [f for f in report.families if f.tier == "exact"]
    members = report_to_dict(report, 20)["families"][0]["members"]
    assert {m.file_path for m in family.members} == {"left.rs", "right.rs"}
    assert {m["file"] for m in members if m["recovered_from_partial_file"]} == {"right.rs"}


def test_broken_function_itself_cannot_become_a_duplicate(tmp_path):
    report = _pair(
        tmp_path, "rs", "fn broken(x:i32) { let y = ; }", "fn broken(x:i32) { let y = ; }"
    )
    assert report.stats["functions"] == 0
    assert not report.families
    assert not report.coverage_complete


@pytest.mark.parametrize(
    "extension,left,right",
    [
        (
            "py",
            "def f(x):\n    return (lambda x: x+1)(x)+x\n",
            "def g(a):\n    return (lambda b: b+1)(a)+a\n",
        ),
        (
            "py",
            "def f(x):\n    return [x+1 for x in x]+[x]\n",
            "def g(a):\n    return [b+1 for b in a]+[a]\n",
        ),
        (
            "py",
            "def f(xs):\n    return [(lambda: item+1)() for item in xs]\n",
            "def g(rows):\n    return [(lambda: value+1)() for value in rows]\n",
        ),
        (
            "js",
            "function f(x){ try{return x();} catch(err){return err.code+1;} }",
            "function g(a){ try{return a();} catch(problem){return problem.code+1;} }",
        ),
        (
            "js",
            "function f(x){ if(x){var y=x+1;} return y+2; }",
            "function g(a){ if(a){var result=a+1;} return result+2; }",
        ),
        (
            "go",
            "package p\nfunc f(x int) int { x,y:=x+1,x+2; return x-y }",
            "package p\nfunc g(a int) int { a,b:=a+1,a+2; return a-b }",
        ),
        (
            "rs",
            "fn f(x:i32)->i32 { let x=x+1; x*3+4 }",
            "fn g(arg:i32)->i32 { let value=arg+1; value*3+4 }",
        ),
        (
            "rs",
            "fn f(x:Option<i32>)->i32 { match x { Some(n)=>n+1, None=>0 } }",
            "fn g(arg:Option<i32>)->i32 { match arg { Some(value)=>value+1, None=>0 } }",
        ),
        (
            "rb",
            "def f(x)\n y=x+1\n y=y+2\n y*3\nend",
            "def g(a)\n result=a+1\n result=result+2\n result*3\nend",
        ),
        (
            "sh",
            'compute() { local value=3; echo "$value"; }',
            'compute() { local value=3; echo "$value"; }',
        ),
    ],
)
def test_scope_boundary_regressions(tmp_path, extension, left, right):
    report = _pair(tmp_path, extension, left, right)
    assert report.coverage_complete, report.stats
    assert any(f.tier == "exact" for f in report.families), report.stats


def test_rust_constructor_identity_is_preserved(tmp_path):
    left = "fn f(x:Option<i32>)->i32 { match x { Some(n)=>n+1, None=>0 } }"
    report = _pair(tmp_path, "rs", left, left.replace("Some(n)", "Other(n)"))
    assert not any(f.tier == "exact" for f in report.families)


def test_python_nested_helper_call_target_is_anchored(tmp_path):
    left = (
        "def f(x):\n def first(y):\n  return y+1\n def second(y):\n  return y*2\n return first(x)\n"
    )
    report = _pair(tmp_path, "py", left, left.replace("return first(x)", "return second(x)"))
    assert not any(
        len(f.members) == 2 and all(m.name == "f" for m in f.members) and f.score == 1
        for f in report.families
    )


def test_python_import_target_is_preserved(tmp_path):
    left = "def f(x):\n from first import helper\n return helper(x)+1\n"
    report = _pair(tmp_path, "py", left, left.replace("from first", "from second"))
    assert not any(f.tier == "exact" for f in report.families)


def test_free_capture_outside_python_unit_retains_identity(tmp_path):
    left = "def outer(first,second):\n def inner(x):\n  return first(x)+1\n return inner\n"
    report = _pair(tmp_path, "py", left, left.replace("return first(x)", "return second(x)"))
    assert not any(
        f.tier == "exact" and any(m.name == "inner" for m in f.members) for f in report.families
    )


def test_same_line_ambiguous_python_scopes_are_reported(tmp_path):
    source = "def f(x):\n return (lambda y: x-y, lambda y: y-x)\n"
    report = _pair(tmp_path, "py", source, source)
    assert not report.coverage_complete
    assert report.stats["binding_error_files"] == 2
    assert not report.families


def test_cpp_out_of_class_method_keeps_its_body(tmp_path):
    left = "int A::compute(int x,int y) { int value=x-y; return value*3+4; }"
    right = "int B::compute(int first,int second) { int result=first-second; return result*3+4; }"
    report = _pair(tmp_path, "cpp", left, right)
    assert report.coverage_complete, report.stats
    assert report.stats["functions"] == 2
    assert any(f.tier == "exact" for f in report.families)


def test_python_exception_binder_rename_is_anchored(tmp_path):
    left = "def f(x):\n try:\n  return compute(x)\n except ValueError as error:\n  return str(error)+str(x)\n"
    right = left.replace("error", "problem")
    report = _pair(tmp_path, "py", left, right)
    assert any(f.tier == "exact" for f in report.families), report.stats


@pytest.mark.parametrize(
    "extension,left,right",
    [
        ("rb", 'def f(x,y)\n "value:#{x}"\nend', 'def f(x,y)\n "value:#{y}"\nend'),
        (
            "kt",
            'fun f(x:Int,y:Int):String { return "value:${x}" }',
            'fun f(x:Int,y:Int):String { return "value:${y}" }',
        ),
        ("sh", 'compute() { echo "$FIRST"; }', 'compute() { echo "$SECOND"; }'),
    ],
)
def test_interpolation_does_not_hide_variable_uses(tmp_path, extension, left, right):
    report = _pair(tmp_path, extension, left, right)
    assert report.stats["functions"] == 2
    assert not any(f.tier == "exact" for f in report.families)


def test_unrecognized_python_compiler_scope_is_partial(tmp_path, monkeypatch):
    from victor.core.graph_rag import clone_bindings

    def unsupported(*args):
        raise ValueError("new compiler scope")

    monkeypatch.setattr(clone_bindings, "python_bindings", unsupported)
    report = _pair(tmp_path, "py", "def f(x):\n return x+1\n", "def f(x):\n return x+1\n")
    assert not report.coverage_complete
    assert report.stats["binding_error_files"] == 2
    assert not report.families


def test_base_install_without_optional_codegraph_still_analyzes_python():
    import subprocess
    import sys

    source = """
import importlib.abc
import sys
import tempfile
from pathlib import Path
class NoCodegraph(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "victor_codegraph" or fullname.startswith("victor_codegraph."):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoCodegraph())
from victor.core.graph_rag.structural_clones import scan
from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES
assert SOURCE_LANGUAGES[".py"] == "python"
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name in ("a.py", "b.py"):
        (root/name).write_text("def f(x):\\n return x+1\\n")
    report = scan(root, min_nodes=5, near_metric="subtree")
    assert report.coverage_complete, report.stats
    assert any(f.tier == "exact" for f in report.families)
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
