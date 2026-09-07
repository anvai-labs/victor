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

"""Build a module-level import graph over the in-repo Python packages and
cluster it to derive module boundaries.

Indexes victor/, victor-contracts/, verticals/, and victor-codegraph/ via AST
import extraction, joins per-module LOC and 12-month git churn, then runs
Louvain community detection at file and subpackage level. Used by the
2026-09-03 co-design review (docs/reviews/2026-09-03-codesign/); kept as a
maintainer script so boundary/cluster analysis is repeatable.

Requires networkx (analysis only, not a runtime dependency).

Outputs (to output/module_graph/, gitignored):
  graph.json     file-level nodes + edges + fan-in/out, LOC, churn, layer
  subpkg.json    subpackage-level graph, clusters, fan-in/out
  clusters.md    human-readable cluster map, cycles (SCCs), churn leaders

Usage:
    python3 scripts/module_graph.py
"""

import ast
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    raise SystemExit("networkx is required: pip install networkx")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output" / "module_graph"
CHURN_SINCE = "1 year ago"

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

CLIENT = {"victor.ui", "victor.integrations"}
FRAMEWORK = {"victor.framework"}
RUNTIME = {"victor.agent", "victor.runtime", "victor.teams", "victor.context"}
INFRA = {
    "victor.providers",
    "victor.tools",
    "victor.state",
    "victor.storage",
    "victor.core",
    "victor.config",
    "victor.native",
    "victor.security",
    "victor.models",
    "victor.deps",
    "victor.protocols",
}


def iter_pkg_roots():
    """Yield (package_name, root_dir) for every in-repo Python package."""
    yield "victor", REPO / "victor"
    for cand in sorted((REPO / "victor-contracts").iterdir()):
        if cand.is_dir() and (cand / "__init__.py").exists() and cand.name[0].isalpha():
            yield cand.name, cand
    for vert in sorted((REPO / "verticals").iterdir()):
        if not vert.is_dir():
            continue
        for base in [vert, vert / "src"]:
            if base.is_dir():
                for cand in sorted(base.iterdir()):
                    if cand.is_dir() and (cand / "__init__.py").exists() and cand.name[0].isalpha():
                        yield cand.name, cand
    for base in list((REPO / "victor-codegraph").glob("*/src")) + [REPO / "victor-codegraph"]:
        if base.is_dir():
            for cand in sorted(base.iterdir()):
                if cand.is_dir() and (cand / "__init__.py").exists() and cand.name[0].isalpha():
                    yield cand.name, cand


def layer_of(mod):
    if mod.startswith("victor.") or mod == "victor":
        top = ".".join(mod.split(".")[:2])
        if top in CLIENT:
            return "client"
        if top in FRAMEWORK:
            return "framework"
        if top in RUNTIME:
            return "runtime"
        if top in INFRA:
            return "infra"
        return "other-victor"
    if mod.startswith(("victor_contracts", "victor_sdk")):
        return "contracts"
    return "vertical/codegraph"


def subpkg(mod):
    if mod.startswith("victor."):
        return ".".join(mod.split(".")[:2])
    return mod.split(".")[0]


def resolve_from(cur_pkg_parts, level, module, names, name_to_path):
    """Return target module names for a from-import, resolved to the deepest
    existing in-repo module."""
    if level == 0:
        base_parts = module.split(".") if module else []
    else:
        base_parts = (
            list(cur_pkg_parts) if level == 1 else cur_pkg_parts[: len(cur_pkg_parts) - (level - 1)]
        )
        if module:
            base_parts += module.split(".")
    if not base_parts:
        return []
    base = None
    for k in range(len(base_parts), 0, -1):
        cand = ".".join(base_parts[:k])
        if cand in name_to_path:
            base = cand
            break
    if base is None:
        return []
    targets = []
    for n in names:
        c = f"{base}.{n}"
        targets.append(c if c in name_to_path else base)
    return targets


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    name_to_path, pkg_roots = {}, {}
    for pkg_name, root in iter_pkg_roots():
        if pkg_name in pkg_roots:
            continue
        pkg_roots[pkg_name] = root
        for f in root.rglob("*.py"):
            if any(p in EXCLUDE_PARTS for p in f.parts):
                continue
            rel = f.relative_to(root).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            mod = ".".join([pkg_name] + parts) if parts else pkg_name
            name_to_path[mod] = f

    loc = {m: len(p.read_text(errors="ignore").splitlines()) for m, p in name_to_path.items()}

    churn = Counter()
    log = subprocess.run(
        ["git", "log", f"--since={CHURN_SINCE}", "--name-only", "--pretty=format:"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout
    for line in log.splitlines():
        if line.strip():
            churn[line.strip()] += 1
    churn_by_mod = {m: churn.get(str(p.relative_to(REPO)), 0) for m, p in name_to_path.items()}

    edges = defaultdict(int)
    for mod, path in sorted(name_to_path.items()):
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        cur_pkg = mod.split(".")[:-1] if path.name != "__init__.py" else mod.split(".")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets = resolve_from(
                    cur_pkg,
                    node.level,
                    node.module,
                    [a.name for a in node.names],
                    name_to_path,
                )
                for t in targets:
                    if t != mod:
                        edges[(mod, t)] += 1
            elif isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    for k in range(len(parts), 0, -1):
                        p = ".".join(parts[:k])
                        if p in name_to_path:
                            if p != mod:
                                edges[(mod, p)] += 1
                            break

    DG = nx.DiGraph()
    DG.add_nodes_from(name_to_path)
    for (s, d), w in edges.items():
        DG.add_edge(s, d, weight=w)

    # file-level clustering (degree-0 nodes excluded)
    UG = DG.to_undirected()
    for u, v, d in UG.edges(data=True):
        d["weight"] = (DG[u][v]["weight"] if DG.has_edge(u, v) else 0) + (
            DG[v][u]["weight"] if DG.has_edge(v, u) else 0
        )
    UG1 = UG.copy()
    UG1.remove_nodes_from([n for n in UG if UG.degree(n) == 0])
    file_comms = sorted(
        nx.community.louvain_communities(UG1, weight="weight", resolution=1.0, seed=42),
        key=len,
        reverse=True,
    )

    # subpackage-level clustering
    sub_of = {m: subpkg(m) for m in name_to_path}
    SG = nx.DiGraph()
    SG.add_nodes_from({sub_of[m] for m in name_to_path})
    for (s, d), w in edges.items():
        a, b = sub_of[s], sub_of[d]
        if a != b:
            SG.add_edge(a, b, weight=SG.get_edge_data(a, b, {}).get("weight", 0) + w)
    SUG = SG.to_undirected()
    for u, v, d in SUG.edges(data=True):
        d["weight"] = (SG[u][v]["weight"] if SG.has_edge(u, v) else 0) + (
            SG[v][u]["weight"] if SG.has_edge(v, u) else 0
        )
    sub_comms = sorted(
        nx.community.louvain_communities(SUG, weight="weight", resolution=0.9, seed=42),
        key=len,
        reverse=True,
    )
    sub_to_comm = {}
    for i, c in enumerate(sub_comms):
        for n in c:
            sub_to_comm[n] = i

    sccs = sorted(
        [s for s in nx.strongly_connected_components(DG) if len(s) > 1],
        key=len,
        reverse=True,
    )
    cross_pkg = Counter()
    for u, v in DG.edges():
        pu, pv = u.split(".")[0], v.split(".")[0]
        if pu != pv:
            cross_pkg[(pu, pv)] += 1

    nodes = [
        {
            "module": m,
            "fan_in": DG.in_degree(m),
            "fan_out": DG.out_degree(m),
            "loc": loc[m],
            "churn": churn_by_mod[m],
            "layer": layer_of(m),
            "subpkg": sub_of[m],
        }
        for m in sorted(name_to_path)
    ]
    (OUT / "graph.json").write_text(
        json.dumps(
            {
                "n_modules": len(name_to_path),
                "n_edges": DG.number_of_edges(),
                "nodes": nodes,
                "edges": [{"s": s, "d": d, "w": w} for (s, d), w in sorted(edges.items())],
                "sccs_gt1": [sorted(s) for s in sccs],
                "cross_package_edges": [
                    {"from": a, "to": b, "count": c} for (a, b), c in cross_pkg.most_common()
                ],
                "pkg_roots": {k: str(v.relative_to(REPO)) for k, v in pkg_roots.items()},
            },
            indent=1,
        )
    )

    sub_nodes = []
    for s in sorted(SG.nodes):
        members = [m for m in name_to_path if sub_of[m] == s]
        sub_nodes.append(
            {
                "subpkg": s,
                "cluster": sub_to_comm[s],
                "modules": len(members),
                "loc": sum(loc[m] for m in members),
                "churn": sum(churn_by_mod[m] for m in members),
                "fan_in": SG.in_degree(s),
                "fan_out": SG.out_degree(s),
                "in_weight": sum(d["weight"] for _, _, d in SG.in_edges(s, data=True)),
                "out_weight": sum(d["weight"] for _, _, d in SG.out_edges(s, data=True)),
                "layer": layer_of(s),
            }
        )
    (OUT / "subpkg.json").write_text(
        json.dumps(
            {
                "clusters": [{"id": i, "members": sorted(c)} for i, c in enumerate(sub_comms)],
                "subpkgs": sub_nodes,
                "edges": [
                    {"s": u, "d": v, "w": d["weight"]} for u, v, d in sorted(SUG.edges(data=True))
                ],
            },
            indent=1,
        )
    )

    L = [
        "# Victor module graph",
        f"modules={len(name_to_path)} edges={DG.number_of_edges()} "
        f"| file communities={len(file_comms)} | subpkg clusters={len(sub_comms)} "
        f"| SCCs>1={len(sccs)} | built in {time.time()-t0:.1f}s",
        "",
        "## Subpackage clusters (resolution 0.9)",
    ]
    for i, c in enumerate(sub_comms):
        L.append(f"### C{i}: {sorted(c)}")
    L += ["", "## Cycles (file-level SCCs > 1), largest"]
    for s in sccs[:15]:
        L.append(f"- {len(s)}: {', '.join(sorted(s)[:10])}")
    L += ["", "## Top fan-in subpackages (stability)"]
    for n in sorted(sub_nodes, key=lambda x: -x["fan_in"])[:25]:
        L.append(
            f"- {n['subpkg']:<28} fanIn={n['fan_in']:>2} fanOut={n['fan_out']:>2} "
            f"churn={n['churn']:>4} C{n['cluster']}"
        )
    L += ["", "## Top churn subpackages (12m)"]
    for n in sorted(sub_nodes, key=lambda x: -x["churn"])[:25]:
        L.append(f"- {n['subpkg']:<28} churn={n['churn']:>4} loc={n['loc']:>7} C{n['cluster']}")
    (OUT / "clusters.md").write_text("\n".join(L))

    print(
        f"modules={len(name_to_path)} edges={DG.number_of_edges()} "
        f"subpkg_clusters={len(sub_comms)} sccs={len(sccs)} -> {OUT} "
        f"in {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
