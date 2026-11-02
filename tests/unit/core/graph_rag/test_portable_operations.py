"""Portable projection retains scope identities, effects and type operations."""

import pytest

from victor.core.graph_rag.structural_clones import scan


def report_pair(root, a, b, suffix="rs"):
    (root / f"a.{suffix}").write_text(a)
    (root / f"b.{suffix}").write_text(b)
    return scan(root, cross_language=True, min_nodes=5, near_metric="subtree", near_threshold=1)


@pytest.mark.parametrize(
    "source",
    [
        "fn f(x:i32)->i32 { let value=crate::utils::compute::<i32>(x); value+1 }",
        "fn f(x:&[i32])->i32 { let value=x[0]; -value+1 }",
        "fn f(x:i32)->Result<i32,Error> { let value=compute(x)?; Ok(value+1) }",
        "fn f(x:i32)->i32 { let values=[x,2,3]; values[0]+1 }",
        "fn f(x:i32)->i64 { let value=x as i64; value+1 }",
        'fn f(x:i32)->String { let value="ready"; format_result(value,x) }',
        "fn f(x:i32)->i32 { let callback=|y| x-y; callback(x)+1 }",
        "fn f(x:Option<i32>)->i32 { match x { Some(n) if n>1=>n*3, _=>2 } }",
    ],
)
def test_rust_operation_coverage(tmp_path, source):
    report = report_pair(tmp_path, source, source)
    assert report.stats["portable_unsupported_units"] == 0, report.stats
    assert report.stats["portable_units"] == 2


@pytest.mark.parametrize(
    "left,right",
    [
        (
            "fn f(x:i32)->i32 { let a=crate::a::call(x); a+1 }",
            "fn f(x:i32)->i32 { let a=crate::b::call(x); a+1 }",
        ),
        (
            "fn f(x:i32)->i32 { let g=|y| y-x; g(x)+1 }",
            "fn f(x:i32)->i32 { let g=|y| x-x; g(x)+1 }",
        ),
        (
            "fn f(x:i32)->i32 { let y=x+1; {let y=x+2; consume(y);} y+3 }",
            "fn f(x:i32)->i32 { let y=x+1; {let z=x+2; consume(y);} y+3 }",
        ),
        ("fn f(x:i32)->i64 { let a=x as i64; a+1 }", "fn f(x:i32)->u64 { let a=x as u64; a+1 }"),
        (
            "fn f(x:i32)->i32 { let a=&x; consume(a)+1 }",
            "fn f(x:i32)->i32 { let a=&mut x; consume(a)+1 }",
        ),
        (
            "fn f(x:i32)->Result<i32,E> { let a=compute(x)?; Ok(a+1) }",
            "fn f(x:i32)->Result<i32,E> { let a=compute(x); Ok(a+1) }",
        ),
    ],
)
def test_portable_negative_controls(tmp_path, left, right):
    report = report_pair(tmp_path, left, right)
    assert report.stats["portable_unsupported_units"] == 0, report.stats
    a, b = report.units
    assert a.portable_hashes != b.portable_hashes


@pytest.mark.parametrize(
    "source",
    [
        "def f(x):\n with lock(x) as guard:\n  value=guard.get(key=x)\n return value+1\n",
        "def f(xs):\n values=[x+1 for x in xs if x>2]\n return values[0]+1\n",
        'def f(x):\n return {"result": x+1, "values": [x,2]}\n',
        "def f(x):\n callback=lambda y:x-y\n return callback(x)+1\n",
        "def f(x):\n try:\n  return compute(x)+1\n except ValueError as error:\n  raise RuntimeError(str(error))\n",
    ],
)
def test_python_operation_coverage(tmp_path, source):
    report = report_pair(tmp_path, source, source, "py")
    assert report.stats["portable_unsupported_units"] == 0, report.stats
    assert report.stats["portable_units"] == 2


def test_real_numeric_semantics_remain_unverified(tmp_path):
    (tmp_path / "a.py").write_text("def f(x):\n return (x+1)*2\n")
    (tmp_path / "b.js").write_text("function f(x) { return (x+1)*2; }")
    report = scan(tmp_path, cross_language=True, min_nodes=5, near_threshold=1)
    assert any(f.tier == "cross_language" for f in report.families)
    assert report.stats["portable_schema"] == "portable-operations-v2"


@pytest.mark.parametrize(
    "source",
    [
        "fn f(x:Option<i32>)->i32 { if let Some(value)=x { return value+1; } 2 }",
        "fn f(rows:Vec<(i32,i32)>)->i32 { let mut value=0; for (a,b) in rows {value+=a+b;} value }",
        "fn f(x:i32)->Value { let value=Value { field:x, x }; value }",
        "fn f(x:i32)->i32 { let mut value=0; for n in 1..=x {value+=n;} value }",
    ],
)
def test_further_rust_operations_keep_complete_projection(tmp_path, source):
    report = report_pair(tmp_path, source, source)
    assert report.stats["portable_unsupported_units"] == 0, report.stats


@pytest.mark.parametrize(
    "left,right",
    [
        (
            "fn f(x:i32)->i32 { for n in 1..x {consume(n);} x+1 }",
            "fn f(x:i32)->i32 { for n in 2..x {consume(n);} x+1 }",
        ),
        (
            "fn f(x:i32)->i32 { for n in 1..x {consume(n);} x+1 }",
            "fn f(x:i32)->i32 { for n in 1..=x {consume(n);} x+1 }",
        ),
        (
            "fn f(x:(i32,i32))->i32 { let (_,a)=x; a+1 }",
            "fn f(x:(i32,i32))->i32 { let (a,_)=x; a+1 }",
        ),
        (
            "fn f(x:i32)->Value { Value { first:x, second:1 } }",
            "fn f(x:i32)->Value { Value { first:1, second:x } }",
        ),
        (
            "fn f(x:Option<i32>,y:i32)->i32 { if let Some(y)=x { return y+1; } y+2 }",
            "fn f(x:Option<i32>,y:i32)->i32 { if let Some(z)=x { return y+1; } y+2 }",
        ),
    ],
)
def test_further_negative_contracts(tmp_path, left, right):
    report = report_pair(tmp_path, left, right)
    assert report.stats["portable_unsupported_units"] == 0, report.stats
    assert report.units[0].portable_hashes != report.units[1].portable_hashes


def test_value_of_rust_if_expression_is_not_erased(tmp_path):
    left = "fn f(x:i32)->i32 { if x>0 { 1 } else { 2 } }"
    right = "fn f(x:i32) { if x>0 { 1 } else { 2 }; }"
    report = report_pair(tmp_path, left, right)
    assert report.stats["portable_unsupported_units"] == 0, report.stats
    assert report.units[0].portable_hashes != report.units[1].portable_hashes
