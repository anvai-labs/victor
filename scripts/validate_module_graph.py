#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTY OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-validate scripts/module_graph.py's AST import graph against
victor_codegraph's IMPORTS edges in .victor/project.db.

Dogfood check: the static AST extraction and the production graph index are
two independent derivations of the same import structure. Agreement means
both are trustworthy; disagreement localizes a bug in one of them. AST-only
edges are expected (they include __init__.py re-exports that the module-node
graph does not represent) — codegraph-only edges are the failure signal.

Run scripts/module_graph.py first.

Usage:
    python3 scripts/validate_module_graph.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GRAPH = REPO / "output" / "module_graph" / "graph.json"
DB = REPO / ".victor" / "project.db"

EXCLUDE_PARTS = {
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "egg-info",
    "node_modules",
    "tests",
    "test",
    "examples",
    "docs",
    "cookbook_tests",
}


def build_file_to_mod(pkg_roots):
    file_to_mod = {}
    for pkg, root in pkg_roots.items():
        for f in root.rglob("*.py"):
            if any(p in EXCLUDE_PARTS for p in f.parts):
                continue
            rel = f.relative_to(root).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            file_to_mod[str(f.relative_to(REPO))] = ".".join([pkg] + parts) if parts else pkg
    return file_to_mod


def main():
    graph = json.loads(GRAPH.read_text())
    ast_edges = {(e["s"], e["d"]) for e in graph["edges"]}
    pkg_roots = {k: REPO / v for k, v in graph["pkg_roots"].items()}
    file_to_mod = build_file_to_mod(pkg_roots)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
    id_to_file = dict(conn.execute("SELECT node_id, file FROM graph_node WHERE type='module'"))
    cg_edges = set()
    unmapped = total = 0
    for src, dst in conn.execute("SELECT src, dst FROM graph_edge WHERE type='IMPORTS'"):
        total += 1
        sm = file_to_mod.get(id_to_file.get(src, ""))
        dm = file_to_mod.get(id_to_file.get(dst, ""))
        if not sm or not dm:
            unmapped += 1
            continue
        cg_edges.add((sm, dm))
    conn.close()

    both = ast_edges & cg_edges
    ast_only = ast_edges - cg_edges
    cg_only = cg_edges - ast_edges

    print(f"AST edges (in-scope pkgs):      {len(ast_edges)}")
    print(
        f"codegraph IMPORTS edges total:  {total} "
        f"(in-scope module-level: {len(cg_edges)}, unmapped: {unmapped})"
    )
    print(f"agreement (intersection):       {len(both)}")
    print(
        f"codegraph-only (FAIL signal):   {len(cg_only)} "
        f"({len(cg_only) / max(len(cg_edges), 1) * 100:.1f}% of codegraph)"
    )
    print(
        f"AST-only (expected, __init__):  {len(ast_only)} "
        f"({len(ast_only) / max(len(ast_edges), 1) * 100:.1f}% of AST)"
    )

    def pkg(m):
        return m.split(".")[0] + (("." + m.split(".")[1]) if m.startswith("victor.") else "")

    for label, bucket in [("codegraph-only", cg_only), ("AST-only", ast_only)]:
        by_pair = defaultdict(int)
        for s, d in bucket:
            by_pair[f"{pkg(s)} -> {pkg(d)}"] += 1
        print(f"\n## {label} by subpackage pair (top 10)")
        for k, v in sorted(by_pair.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k}: {v}")
        print(f"## {label} sample (10)")
        for s, d in sorted(bucket)[:10]:
            print(f"  {s} -> {d}")

    if cg_only:
        print(
            "\nRESULT: FAIL — codegraph reports imports the AST graph lacks; "
            "investigate before trusting either."
        )
        raise SystemExit(1)
    print("\nRESULT: PASS — codegraph IMPORTS are a subset of the AST graph.")


if __name__ == "__main__":
    main()
