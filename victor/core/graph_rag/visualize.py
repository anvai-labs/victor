# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Hierarchical coupling visualizations rendered from the indexed graph.

Self-analysis round 2 (2026-10-23): the ad-hoc package-coupling view that
quantified the structural findings, productized as three levels —
``package`` (cross-package coupling, SCC members highlighted),
``module`` (file coupling inside one package) and ``class`` (symbol graph
inside one module). Pure query -> mermaid-text functions over the project's
SQLite graph store; deterministic (sorted) output so golden tests can pin
it. Rendering to SVG is the CLI's job (``mmdc`` when present) — these
functions never shell out.

Scope: the SQLite project store (``.victor/project.db``). ProximaDB-backed
repositories are out of scope for v1 — the store protocol cannot express
the GROUP-BY aggregations these views need.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_EDGE_TYPES = ("CALLS", "REFERENCES")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _package_of(file_path: str) -> Optional[str]:
    """Map a repo-relative file to its package (first two components for
    first-party trees, else the top directory)."""
    parts = file_path.split("/")
    if (
        parts[0] in ("victor", "verticals", "victor-contracts", "victor-codegraph")
        and len(parts) > 2
    ):
        return "/".join(parts[:2])
    if len(parts) > 1:
        return parts[0]
    return None


def _file_package_of(file_path: str) -> Optional[str]:
    return _package_of(file_path) if "/" in file_path else None


def _mermaid_id(label: str) -> str:
    return label.replace("/", "_").replace(".", "_").replace("-", "_")


def _tarjan_sccs(adjacency: Dict[str, Set[str]]) -> List[List[str]]:
    """Iterative Tarjan — the aggregate cycle islands to highlight."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    sccs: List[List[str]] = []
    counter = 0

    for root in sorted(adjacency):
        if root in index:
            continue
        work = [(root, iter(sorted(adjacency[root])))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(adjacency[child]))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                sccs.append(sorted(component))
    return sccs


def _cross_target_edges(conn: sqlite3.Connection) -> List[Tuple[str, str, int]]:
    """Aggregated src-package -> dst-package weights over CALLS/REFERENCES."""
    weights: Dict[Tuple[str, str], int] = defaultdict(int)
    rows = conn.execute(
        f"""
        SELECT ns.file AS src_file, nd.file AS dst_file
        FROM graph_edge e
        JOIN graph_node ns ON ns.node_id = e.src
        JOIN graph_node nd ON nd.node_id = e.dst
        WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
          AND ns.file != nd.file
        """,
        _EDGE_TYPES,
    )
    for row in rows:
        src, dst = _file_package_of(row["src_file"]), _file_package_of(row["dst_file"])
        if src and dst and src != dst:
            weights[(src, dst)] += 1
    return sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))


def package_coupling_mermaid(db_path: Path, top_packages: int = 12) -> str:
    """Cross-package coupling (compiler-verified where tier 3 ran).

    CALLS/REFERENCES edges aggregated to package level; the heaviest
    packages and their strongest couplings; multi-package cycle islands
    (SCCs) highlighted.
    """
    conn = _connect(db_path)
    try:
        ranked = _cross_target_edges(conn)
    finally:
        conn.close()

    packages: List[str] = []
    for (src, dst), _ in ranked:
        for candidate in (src, dst):
            if candidate not in packages:
                packages.append(candidate)
        if len(packages) >= top_packages:
            break
    top = set(packages)

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    degree: Dict[str, int] = defaultdict(int)
    edges: List[Tuple[str, str, int]] = []
    for (src, dst), weight in ranked:
        adjacency[src].add(dst)
        if src in top:
            degree[src] += weight
        if dst in top:
            degree[dst] += weight
        if src in top and dst in top and len(edges) < 24:
            edges.append((src, dst, weight))

    cycle_members: Set[str] = set()
    for component in _tarjan_sccs(adjacency):
        if len(component) > 1:
            cycle_members.update(component)

    lines = [
        "%% Cross-package coupling - CALLS/REFERENCES edges (compiler-provenance where tier 3 ran)",
        "graph LR",
        "  classDef cycle fill:#fff1f2,stroke:#e11d48,color:#4c0519,stroke-width:2.5px",
        "  classDef heavy fill:#fffbeb,stroke:#d97706,color:#451a03",
    ]
    for package in packages:
        label = package.split("/")[-1]
        node_class = "cycle" if package in cycle_members else "heavy"
        lines.append(f'  {_mermaid_id(package)}["{label}"]:::{node_class}')
    for src, dst, weight in edges:
        lines.append(f"  {_mermaid_id(src)} --{weight}--> {_mermaid_id(dst)}")
    if cycle_members & top:
        lines.append("  %% red nodes: members of a cross-package import/call cycle island")
    return "\n".join(lines) + "\n"


def module_coupling_mermaid(db_path: Path, package: str, top_modules: int = 12) -> str:
    """File-level coupling inside one package (prefix match on the path)."""
    conn = _connect(db_path)
    try:
        weights: Dict[Tuple[str, str], int] = defaultdict(int)
        rows = conn.execute(
            f"""
            SELECT ns.file AS src_file, nd.file AS dst_file
            FROM graph_edge e
            JOIN graph_node ns ON ns.node_id = e.src
            JOIN graph_node nd ON nd.node_id = e.dst
            WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
              AND ns.file != nd.file
              AND ns.file LIKE ?
              AND nd.file LIKE ?
            """,
            (*_EDGE_TYPES, f"{package}/%", f"{package}/%"),
        )
        for row in rows:
            weights[(row["src_file"], row["dst_file"])] += 1
    finally:
        conn.close()

    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    modules: List[str] = []
    for (src, dst), _ in ranked:
        for candidate in (src, dst):
            if candidate not in modules:
                modules.append(candidate)
        if len(modules) >= top_modules:
            break
    top = set(modules)

    lines = [
        f"%% Module coupling inside {package} - CALLS/REFERENCES edges",
        "graph LR",
        "  classDef heavy fill:#fffbeb,stroke:#d97706,color:#451e03",
    ]
    for module_path in modules:
        label = module_path.rsplit("/", 1)[-1]
        lines.append(f'  {_mermaid_id(module_path)}["{label}"]:::heavy')
    for (src, dst), weight in ranked:
        if src in top and dst in top:
            lines.append(f"  {_mermaid_id(src)} --{weight}--> {_mermaid_id(dst)}")
    return "\n".join(lines) + "\n"


def class_graph_mermaid(db_path: Path, module_file: str, top_symbols: int = 16) -> str:
    """Symbol-level graph for one module: nodes by degree over intra-module
    CALLS/REFERENCES edges (tier-3 compiler edges dominate when present)."""
    conn = _connect(db_path)
    try:
        nodes = {
            row["node_id"]: (row["name"], row["type"])
            for row in conn.execute(
                "SELECT node_id, name, type FROM graph_node WHERE file = ? AND type != 'statement'",
                (module_file,),
            )
        }
        weights: Dict[Tuple[str, str], int] = defaultdict(int)
        for row in conn.execute(
            f"""
            SELECT e.src, e.dst FROM graph_edge e
            JOIN graph_node ns ON ns.node_id = e.src
            JOIN graph_node nd ON nd.node_id = e.dst
            WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
              AND ns.file = ? AND nd.file = ?
            """,
            (*_EDGE_TYPES, module_file, module_file),
        ):
            weights[(row["src"], row["dst"])] += 1
    finally:
        conn.close()

    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    degree: Dict[str, int] = defaultdict(int)
    for (src, dst), weight in ranked:
        degree[src] += weight
        degree[dst] += weight
    keep: Set[str] = set()
    for node_id, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0])):
        keep.add(node_id)
        if len(keep) >= top_symbols:
            break

    lines = [
        f"%% Symbol graph for {module_file} - intra-module CALLS/REFERENCES",
        "graph LR",
        "  classDef fn fill:#ecfeff,stroke:#0891b2,color:#0f172a",
        "  classDef type fill:#f0fdf4,stroke:#16a34a,color:#052e16",
    ]
    for node_id in sorted(keep):
        name, node_type = nodes.get(node_id, (node_id, "function"))
        node_class = "type" if node_type in ("class", "struct", "interface", "enum") else "fn"
        lines.append(f'  {_mermaid_id(node_id)}["{name}"]:::{node_class}')
    for (src, dst), weight in ranked:
        if src in keep and dst in keep:
            lines.append(f"  {_mermaid_id(src)} --{weight}--> {_mermaid_id(dst)}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class VizResult:
    """A rendered view: the mermaid text plus the output paths written."""

    mermaid: str
    mmd_path: Path
    svg_path: Optional[Path] = None


def render(mermaid: str, out_path: Path) -> VizResult:
    """Write the .mmd and (when mmdc is on PATH) render the .svg beside it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mermaid, encoding="utf-8")
    svg_path: Optional[Path] = None
    import shutil
    import subprocess

    mmdc = shutil.which("mmdc")
    if mmdc:
        svg_target = out_path.with_suffix(".svg")
        completed = subprocess.run(  # noqa: S603 - fixed argv from shutil.which
            [mmdc, "-i", str(out_path), "-o", str(svg_target), "-b", "white", "-w", "1600"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode == 0 and svg_target.exists():
            svg_path = svg_target
    return VizResult(mermaid=mermaid, mmd_path=out_path, svg_path=svg_path)
