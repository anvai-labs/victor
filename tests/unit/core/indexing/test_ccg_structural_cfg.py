# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structural CFG edges in the built-in CCG builder (closes the old TODO).

Pins AST-derived control flow on top of the sequential heuristic: if/else
false branches, loop back-edges, try/except/finally exception flow, and
switch/match case dispatch. The enhanced (plugin) builder is bypassed so
these tests exercise the built-in path that core-only installs rely on.
"""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")

from victor.core.indexing.ccg_builder import CodeContextGraphBuilder  # noqa: E402


async def _build(tmp_path: Path, source: str, name: str = "m.py", language: str = "python"):
    f = tmp_path / name
    f.write_text(source)
    builder = CodeContextGraphBuilder(language=language)
    builder._resolve_enhanced_builder = lambda lang: None  # force built-in path
    nodes, edges = await builder.build_ccg_for_file(f, language)
    lines = {n.node_id: n.line for n in nodes}

    def typed(edge_type: str):
        return sorted(
            (lines.get(e.src), lines.get(e.dst))
            for e in edges
            if getattr(e.type, "value", str(e.type)) == edge_type
        )

    return typed


PY = """\
def process(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
        else:
            total -= item
    try:
        risky()
    except ValueError:
        handle()
    finally:
        cleanup()
    match total:
        case 0:
            return None
        case _:
            return total
"""


@pytest.mark.asyncio
async def test_python_if_else_false_branch(tmp_path):
    typed = await _build(tmp_path, PY)
    # if (line 4) -> first statement of the else body (line 7)
    assert (4, 7) in typed("CFG_FALSE")


@pytest.mark.asyncio
async def test_python_loop_back_edge(tmp_path):
    typed = await _build(tmp_path, PY)
    # last statement in the for body (line 7) -> loop header (line 3)
    assert (7, 3) in typed("CFG_LOOP_BACK")


@pytest.mark.asyncio
async def test_python_exception_flow(tmp_path):
    typed = await _build(tmp_path, PY)
    assert (8, 10) in typed("CFG_CATCH")  # try -> except
    assert (8, 12) in typed("CFG_FINALLY")  # try -> finally


@pytest.mark.asyncio
async def test_python_match_case_dispatch(tmp_path):
    typed = await _build(tmp_path, PY)
    cases = typed("CFG_CASE")
    assert (14, 15) in cases  # match -> case 0
    assert (14, 17) in cases  # match -> case _


@pytest.mark.asyncio
async def test_python_no_structural_edges_for_straightline_code(tmp_path):
    typed = await _build(tmp_path, "def f():\n    a = 1\n    b = 2\n    return a + b\n")
    for edge_type in ("CFG_FALSE", "CFG_LOOP_BACK", "CFG_CATCH", "CFG_FINALLY", "CFG_DEFAULT"):
        assert typed(edge_type) == []


JS = """\
function handle(x) {
  while (x > 0) {
    x -= 1;
  }
  try {
    risky();
  } catch (e) {
    recover();
  }
  switch (x) {
    case 0:
      return null;
    default:
      return x;
  }
}
"""


@pytest.mark.asyncio
async def test_javascript_structural_edges(tmp_path):
    pytest.importorskip("tree_sitter_javascript")
    typed = await _build(tmp_path, JS, name="m.js", language="javascript")

    assert (3, 2) in typed("CFG_LOOP_BACK")  # while body -> header
    assert (5, 7) in typed("CFG_CATCH")  # try -> catch
    assert (10, 11) in typed("CFG_CASE")  # switch -> case 0
    assert (10, 13) in typed("CFG_DEFAULT")  # switch -> default


@pytest.mark.asyncio
async def test_structural_edges_deduped_against_sequential(tmp_path):
    """Structural pass must not duplicate (src, dst, type) triples."""
    f = tmp_path / "m.py"
    f.write_text(PY)
    builder = CodeContextGraphBuilder(language="python")
    builder._resolve_enhanced_builder = lambda lang: None
    _nodes, edges = await builder.build_ccg_for_file(f, "python")
    cfg = [
        (e.src, e.dst, getattr(e.type, "value", str(e.type)))
        for e in edges
        if getattr(e.type, "value", str(e.type)).startswith("CFG_")
    ]
    assert len(cfg) == len(set(cfg))
