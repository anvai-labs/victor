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

"""JSON graph queries for the interactive explorer (ADR-033).

The mermaid layer (``visualize.py``) renders static snapshots; this module
answers the SAME aggregations as Cytoscape-ready element lists (the shape
``graph_export.py`` established for workflow graphs) so the web explorer
can navigate package -> module -> class as zoom states of one graph, plus
the diagram families (phase 2: ``flow`` call sequences and
``class_diagram`` type neighborhoods) as mermaid sources generated
Python-side — one grammar shared with the static export. Deterministic
ordering everywhere (weight desc, name asc) — golden tests pin it. Same
scope as visualize.py: the SQLite project store; the store protocol
cannot express the GROUP-BY aggregations.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional

from victor.core.graph_rag.visualize import (
    FFI_IPC_BOUNDARIES,
    _file_edge_weights,
    _package_of,
)

_EDGE_TYPES = ("CALLS", "REFERENCES")
# Statement nodes are CCG machinery (97% of upserts, FTS-skipped): the
# explorer shows symbols, not statement soup.
_SELECT_NODE_TYPES = ("class", "function", "method", "module", "struct", "interface", "enum")


def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _elements(
    nodes: Dict[str, Dict[str, Any]],
    edges: Dict[tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Render the internal maps to the Cytoscape elements shape."""
    return {
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges, key=lambda e: (e[0], e[1], e[2]))],
    }


def _node_entry(node_id: str, label: str, **data: Any) -> Dict[str, Any]:
    return {"data": {"id": node_id, "label": label, **data}}


def _edge_entry(src: str, dst: str, kind: str, weight: int) -> Dict[str, Any]:
    return {
        "data": {
            "id": f"{src}->{dst}:{kind}",
            "source": src,
            "target": dst,
            "kind": kind,
            "weight": weight,
        }
    }


# Slots reserved beyond the weight fill: unrepresented-tree admission
# (vscode-victor), cross-edge partner pull-in (rust crates), then weight
# fill — in that order, each with a floor for the stages after it. The
# budget SCALES with the view size (a fixed 8 starved partners on this
# repo: python-bindings dropped out); the canvas is zoomable and the
# island envelopes keep groups navigable, so breadth beats scarcity.


def package_graph(db_path, top_packages: int = 12) -> Dict[str, List[Dict[str, Any]]]:
    """Cross-package coupling as Cytoscape elements (weight = CALLS/REFERENCES
    count). Nodes carry degree; edges carry weight."""
    conn = _connect(db_path)
    try:
        ranked = _file_edge_weights(conn, key_mapper=_package_of)
    finally:
        conn.close()

    # Architectural coverage first: every top-level TREE of the monorepo
    # is represented (its highest-coupled package), then remaining slots by
    # weight. Without this, small-but-first-class trees (rust, vscode-victor)
    # drown under python's coupling and never appear — the monorepo's other
    # languages look unharvested when the graph has them all along.
    packages: List[str] = []

    def tree_of(pkg: str) -> str:
        return pkg.split("/", 1)[0]

    best_of_tree: Dict[str, str] = {}
    seen_trees = set()
    tree_order: List[str] = []
    for (src, dst), _ in ranked:
        for candidate in (src, dst):
            tree = tree_of(candidate)
            if tree not in best_of_tree:
                best_of_tree[tree] = candidate
                tree_order.append(tree)
    for tree in tree_order:
        rep = best_of_tree[tree]
        if rep not in packages:
            packages.append(rep)
            seen_trees.add(tree)
    for (src, dst), _ in ranked:
        for candidate in (src, dst):
            if len(packages) >= top_packages + len(seen_trees):
                break
            if candidate not in packages:
                packages.append(candidate)
        if len(packages) >= top_packages + len(seen_trees):
            break
    # Trees whose coupling is entirely INTERNAL (vscode-victor: one
    # extension, no cross-package edges) never appear in a cross-edge
    # ranking. Guarantee each unrepresented TREE admission by total
    # activity first — polyglot trees are architectural peers, not
    # leftovers — then fill remaining capacity by weight alone.
    conn = _connect(db_path)
    try:
        internal_all = _internal_activity_packages(conn, exclude=set(packages))
    finally:
        conn.close()
    admitted: Dict[str, int] = {}
    # Dedicated polyglot budget: the weight fill consumes the base slots,
    # so tree guarantees and partner pull-in need their OWN margin or the
    # monorepo's other languages starve again (measured: capacity hit 0).
    # Scales with the requested view size.
    polyglot_margin = max(8, top_packages)
    fill_budget = max(3, top_packages // 4)
    partner_budget = max(3, top_packages // 4)
    capacity = polyglot_margin
    internal_by_tree: Dict[str, tuple] = {}
    for package, weight in internal_all:
        tree = package.split("/", 1)[0]
        if tree not in internal_by_tree:
            internal_by_tree[tree] = (package, weight)
    for tree, (package, weight) in sorted(internal_by_tree.items()):
        if tree in seen_trees or capacity <= partner_budget:
            continue
        packages.append(package)
        admitted[package] = weight
        seen_trees.add(tree)
        capacity -= 1
    top = set(packages)
    # FFI/IPC boundary endpoints: both sides must render for the boundary
    # edge to appear — pull in missing endpoints before partner pull-in.
    for boundary in FFI_IPC_BOUNDARIES:
        for endpoint in (boundary["from"], boundary["to"]):
            if endpoint not in top and capacity > fill_budget:
                packages.append(endpoint)
                top.add(endpoint)
                capacity -= 1
    # Partner pull-in BEFORE weight fill: a cross-edge with exactly ONE
    # endpoint selected dangles (rust crates couple cross-crate; one tree
    # rep hides the rest). Grow along real coupling first — weight fill
    # would otherwise consume the whole margin on already-visible trees.
    # Tree-fair pull-in: one partner PER TREE per pass. Weight order
    # starves light trees — python's one-sided pairs outweigh rust's
    # featherweight cross-crate couplings (26 edges) and eat the budget
    # before any crate partner lands.
    while capacity > fill_budget:
        best_by_tree: Dict[str, tuple] = {}
        for pair, weight in ranked:
            src, dst = pair
            if (src in top) != (dst in top):
                partner = dst if src in top else src
                tree = partner.split("/", 1)[0]
                if partner not in top and tree not in best_by_tree:
                    best_by_tree[tree] = (partner, weight)
        if not best_by_tree:
            break
        for tree in sorted(best_by_tree):
            if capacity <= fill_budget:
                break
            partner, _weight = best_by_tree[tree]
            packages.append(partner)
            top.add(partner)
            capacity -= 1
    for package, weight in internal_all:
        if capacity <= 0:
            break
        if package not in packages:
            packages.append(package)
            admitted[package] = weight
            capacity -= 1

    degree: Dict[str, int] = defaultdict(int)

    def _package_label(pkg: str) -> str:
        # Last component — unless it is a GENERIC directory name (src, lib,
        # crates): 'vscode-victor/src' must not display as just 'src'.
        last = pkg.rsplit("/", 1)[-1]
        if last in ("src", "lib", "crates", "tests", "unit", "core"):
            return pkg
        return last

    nodes: Dict[str, Dict[str, Any]] = {
        p: _node_entry(p, _package_label(p), kind="package") for p in packages
    }
    edges: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for (src, dst), weight in ranked:
        if src in top and dst in top:
            degree[src] += weight
            degree[dst] += weight
            edges[(src, dst, "coupling")] = _edge_entry(src, dst, "coupling", weight)
    # Cross-language FFI/IPC boundaries: declarative architectural edges
    # (the per-language graphs cannot see across the bridge). Weight from
    # each boundary's weight_query against the graph.
    for boundary in FFI_IPC_BOUNDARIES:
        bsrc, bdst = boundary["from"], boundary["to"]
        if bsrc in top and bdst in top:
            try:
                bconn = _connect(db_path)
                try:
                    weight = max(1, bconn.execute(boundary["weight_query"]).fetchone()[0])
                finally:
                    bconn.close()
            except Exception:
                weight = 1
            edges[(bsrc, bdst, boundary["kind"])] = _edge_entry(
                bsrc, bdst, boundary["kind"], weight
            )
            degree[bsrc] += weight
            degree[bdst] += weight

    conn = _connect(db_path)
    try:
        internal_all = _internal_activity_packages(conn, exclude=set())
    finally:
        conn.close()
    internal_weight = dict(internal_all)
    for package in packages:
        # Cross-coupled packages read their cross-edge degree; internally-
        # coupled packages (one-crate/one-extension trees) carry their
        # internal weight — an honest 'couples within itself' number.
        nodes[package]["data"]["degree"] = max(
            degree.get(package, 0), internal_weight.get(package, 0)
        )
    return _elements(nodes, edges)


def _internal_activity_packages(conn: sqlite3.Connection, *, exclude: set) -> List[tuple]:
    """Packages by TOTAL CALLS/REFERENCES involvement (internal + cross) —
    the admission path for trees whose coupling never crosses a package
    boundary (one-crate/one-extension trees)."""
    weights: Dict[str, int] = defaultdict(int)
    for row in conn.execute(
        f"""
        SELECT s.file AS sfile, d.file AS dfile
        FROM graph_edge e
        JOIN graph_node s ON s.node_id = e.src
        JOIN graph_node d ON d.node_id = e.dst
        WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
        """,
        _EDGE_TYPES,
    ):
        for f in (row["sfile"], row["dfile"]):
            pkg = _package_of(f)
            if pkg:
                weights[pkg] += 1
    ranked_internal = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(p, w) for p, w in ranked_internal if p not in exclude]


def module_graph(db_path, package: str, top_modules: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """File-level coupling inside one package as Cytoscape elements."""
    conn = _connect(db_path)
    try:
        prefix = f"{package}/%"
        ranked = _file_edge_weights(conn, path_filter=prefix, key_mapper=lambda p: p or None)
        node_rows = conn.execute(
            "SELECT DISTINCT file FROM graph_node WHERE file LIKE ?",
            (prefix,),
        ).fetchall()
        file_kinds: Dict[str, str] = {row["file"]: "module" for row in node_rows}
    finally:
        conn.close()

    modules: List[str] = []
    for (src, dst), _ in ranked:
        for candidate in (src, dst):
            if candidate not in modules:
                modules.append(candidate)
        if len(modules) >= top_modules:
            break
    top = set(modules)

    nodes: Dict[str, Dict[str, Any]] = {}
    for file_path in sorted(file_kinds):
        if file_path in top or not top:
            nodes[file_path] = _node_entry(file_path, file_path.rsplit("/", 1)[-1], kind="module")
    edges: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for (src, dst), weight in ranked:
        if src in top and dst in top:
            edges[(src, dst, "coupling")] = _edge_entry(src, dst, "coupling", weight)
    return _elements(nodes, edges)


def class_graph(
    db_path, module_file: str, top_symbols: int = 40
) -> Dict[str, List[Dict[str, Any]]]:
    """Symbol-level elements for one module (symbols + their intra-module
    edges), statement nodes excluded."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT node_id, name, type, line, metadata
            FROM graph_node
            WHERE file = ?
              AND node_id IN (
                SELECT node_id FROM graph_node
                WHERE file = ? AND type IN ({", ".join("?" * len(_SELECT_NODE_TYPES))})
              )
            ORDER BY line, name
            """,
            (module_file, module_file, *_SELECT_NODE_TYPES),
        ).fetchall()
        node_ids = {row["node_id"] for row in rows}
        edges: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for row in conn.execute(
            f"""
            SELECT e.src, e.dst, e.type, e.metadata
            FROM graph_edge e
            JOIN graph_node ns ON ns.node_id = e.src
            JOIN graph_node nd ON nd.node_id = e.dst
            WHERE e.type IN ({", ".join("?" * len(_EDGE_TYPES))})
              AND ns.file = ? AND nd.file = ?
            """,
            (*_EDGE_TYPES, module_file, module_file),
        ):
            if row["src"] in node_ids and row["dst"] in node_ids:
                key = (row["src"], row["dst"], row["type"])
                edges[key] = _edge_entry(row["src"], row["dst"], row["type"], 1)
    finally:
        conn.close()

    # Degree-ranked cap: keep the top symbols by edge involvement.
    degree: Dict[str, int] = defaultdict(int)
    for src, dst, _ in edges:
        degree[src] += 1
        degree[dst] += 1
    keep = {nid for nid, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:top_symbols]}
    keep.update(list(node_ids)[: max(0, top_symbols - len(keep))])

    nodes: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["node_id"] in keep:
            nodes[row["node_id"]] = _node_entry(
                row["node_id"], row["name"], kind=row["type"], line=row["line"]
            )
    edges = {key: entry for key, entry in edges.items() if key[0] in keep and key[1] in keep}
    return _elements(nodes, edges)


def search(db_path, query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Symbol search results as node entries (name match, statement-free)."""
    conn = _connect(db_path)
    try:
        like = f"%{query}%"
        placeholders = ",".join("?" for _ in _SELECT_NODE_TYPES)
        rows = conn.execute(
            f"""
            SELECT node_id, name, type, file, line
            FROM graph_node
            WHERE name LIKE ?
              AND type IN ({placeholders})
            ORDER BY name, file, line
            LIMIT ?
            """,
            (like, *_SELECT_NODE_TYPES, limit),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "data": {
                "id": row["node_id"],
                "label": row["name"],
                "kind": row["type"],
                "file": row["file"],
                "line": row["line"],
            }
        }
        for row in rows
    ]


# --- Call-flow sequence diagrams (ADR-033 phase 2) -------------------------
#
# The explorer's zoom levels answer "what is here"; flow answers "how does a
# request move through it". The carrier is CALLS edges; message order within a
# caller is the persisted ``call_line`` (first-use site, 1-based). Graphs
# indexed before that metadata existed fall back to a deterministic but
# unordered listing — surfaced honestly as ``ordered: false``, never guessed.

_FLOW_ENTRY_TYPES = ("function", "method", "constructor")
_FLOW_MAX_DEPTH = 5
_MERMAID_UNSAFE = str.maketrans(
    {'"': "'", "\n": " ", "\r": " ", ":": " -", "<": "", ">": "", "&": "+"}
)


def _mermaid_safe(text: str) -> str:
    """Strip characters that break a mermaid sequenceDiagram token."""
    return str(text).translate(_MERMAID_UNSAFE).strip()


def flow(
    db_path,
    symbol: str,
    *,
    node_id: Optional[str] = None,
    depth: int = 2,
    granularity: str = "class",
    max_messages: int = 150,
    max_fanout: int = 12,
) -> Dict[str, Any]:
    """Call-flow view for one entry symbol (ADR-033 sequence diagrams).

    BFS over CALLS edges, expanded once per symbol (cycles become revisit
    notes, not loops). Messages within a caller are ordered by the persisted
    ``call_line``; when any edge predates that metadata the response is
    flagged ``ordered: false`` and falls back to name order. Participants
    aggregate callees by owning class (methods/constructors) or module
    (functions / class granularity misses). Truncation under the
    ``max_messages``/``max_fanout`` caps is reported, never silent.

    ``node_id`` pins the entry when the caller already disambiguated (the
    explorer's inspector knows it). Returns ``{"entry": None}`` when the
    symbol is unknown and ``{"candidates": [...]}`` when the simple name
    matches several symbols.
    """
    depth = max(1, min(depth, _FLOW_MAX_DEPTH))
    granularity = granularity if granularity in ("class", "module") else "class"
    conn = _connect(db_path)
    try:
        if node_id:
            placeholders = ", ".join("?" for _ in _FLOW_ENTRY_TYPES)
            rows = conn.execute(
                f"""
                SELECT node_id, name, type, file, line FROM graph_node
                WHERE node_id = ? AND type IN ({placeholders})
                """,
                (node_id, *_FLOW_ENTRY_TYPES),
            ).fetchall()
            entries = [dict(row) for row in rows]
        else:
            entries = _flow_entry_candidates(conn, symbol)
        if not entries:
            return {"entry": None}
        if len(entries) > 1:
            return {"candidates": entries}
        entry = entries[0]

        nodes: Dict[str, Dict[str, Any]] = {entry["node_id"]: entry}
        messages: List[Dict[str, Any]] = []
        ordered = True
        truncated = False
        pruned = 0
        visited = {entry["node_id"]}
        frontier = [entry["node_id"]]

        for level in range(depth):
            callees = _flow_callees(conn, frontier)
            next_frontier: List[str] = []
            for caller_id in frontier:
                outgoing = callees.get(caller_id, [])
                if any(row["call_line"] is None for row in outgoing):
                    ordered = False
                if len(outgoing) > max_fanout:
                    truncated = True
                    pruned += len(outgoing) - max_fanout
                    outgoing = outgoing[:max_fanout]
                for row in outgoing:
                    if len(messages) >= max_messages:
                        truncated = True
                        pruned += 1
                        continue
                    dst_id = row["dst"]
                    revisit = dst_id in visited
                    if not revisit:
                        visited.add(dst_id)
                        nodes[dst_id] = row
                        next_frontier.append(dst_id)
                    label = (
                        f"\u00abnew\u00bb {row['name']}" if row["type"] == "class" else row["name"]
                    )
                    messages.append(
                        {
                            "from": caller_id,
                            "to": dst_id,
                            "label": label,
                            "call_line": row["call_line"],
                            "depth": level + 1,
                            "revisit": revisit,
                        }
                    )
            if not next_frontier:
                break
            frontier = next_frontier

        participants = _flow_participants(conn, nodes, granularity)
        return {
            "entry": {k: entry[k] for k in ("node_id", "name", "type", "file", "line")},
            "granularity": granularity,
            "depth": depth,
            "ordered": ordered,
            "truncated": truncated,
            "pruned": pruned,
            "participants": participants,
            "messages": [
                {
                    "from_participant": participants[caller_id]["label"],
                    "to_participant": participants[msg["to"]]["label"],
                    "label": msg["label"],
                    "call_line": msg["call_line"],
                    "depth": msg["depth"],
                    "revisit": msg["revisit"],
                }
                for msg in messages
                if msg["from"] in participants and msg["to"] in participants
            ],
            "mermaid": _flow_mermaid(participants, messages, truncated, pruned),
        }
    finally:
        conn.close()


def _flow_entry_candidates(conn: sqlite3.Connection, symbol: str) -> List[Dict[str, Any]]:
    placeholders = ", ".join("?" for _ in _FLOW_ENTRY_TYPES)
    rows = conn.execute(
        f"""
        SELECT node_id, name, type, file, line
        FROM graph_node
        WHERE name = ? AND type IN ({placeholders})
        ORDER BY file, line, node_id
        """,
        (symbol, *_FLOW_ENTRY_TYPES),
    ).fetchall()
    return [dict(row) for row in rows]


def _flow_callees(conn: sqlite3.Connection, frontier: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Outgoing CALLS per caller, deduped per (callee, line), line-ordered."""
    callees: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    placeholders = ", ".join("?" for _ in frontier)
    seen: Dict[str, set] = defaultdict(set)
    for row in conn.execute(
        f"""
        SELECT e.src, e.dst, e.metadata, n.name, n.type, n.file, n.line
        FROM graph_edge e
        JOIN graph_node n ON n.node_id = e.dst
        WHERE e.type = 'CALLS' AND e.src IN ({placeholders})
        """,
        tuple(frontier),
    ):
        call_line = None
        if row["metadata"]:
            try:
                value = json.loads(row["metadata"]).get("call_line")
            except (ValueError, TypeError):
                value = None
            if isinstance(value, int) and value > 0:
                call_line = value
        key = (row["dst"], call_line)
        if key in seen[row["src"]]:
            continue
        seen[row["src"]].add(key)
        callees[row["src"]].append(
            {
                "dst": row["dst"],
                "name": row["name"],
                "type": row["type"],
                "file": row["file"],
                "line": row["line"],
                "call_line": call_line,
            }
        )
    for rows in callees.values():
        # Program order when call lines are known; otherwise a stable,
        # clearly-flagged name fallback (ordered: false in the response).
        rows.sort(key=lambda r: (r["call_line"] is None, r["call_line"] or 0, r["name"], r["dst"]))
    return callees


def _flow_participants(
    conn: sqlite3.Connection,
    nodes: Dict[str, Dict[str, Any]],
    granularity: str,
) -> Dict[str, Dict[str, Any]]:
    """Map every symbol in the flow onto a sequence-diagram participant.

    Class granularity: methods/constructors join their owning class's lane;
    functions (and class targets of constructor calls) get a module lane.
    """
    placeholders = ", ".join("?" for _ in nodes)
    parent_of = {
        row["node_id"]: row["parent_id"]
        for row in conn.execute(
            f"SELECT node_id, parent_id FROM graph_node WHERE node_id IN ({placeholders})",
            tuple(nodes),
        )
    }
    parent_ids = {pid for pid in parent_of.values() if pid is not None}
    parents = (
        {
            row["node_id"]: dict(row)
            for row in conn.execute(
                f"""
            SELECT node_id, name, type, file FROM graph_node
            WHERE node_id IN ({", ".join("?" for _ in parent_ids)})
            """,
                tuple(parent_ids),
            )
        }
        if parent_ids
        else {}
    )

    participants: Dict[str, Dict[str, Any]] = {}
    for node_id, node in nodes.items():
        parent = parents.get(parent_of.get(node_id))
        if node["type"] == "class":
            # Constructor-call target (method -> class CALLS edge): the class
            # IS the interaction participant.
            participants[node_id] = {
                "key": node_id,
                "label": node["name"],
                "kind": "class",
                "file": node["file"],
            }
            continue
        if granularity == "class" and node["type"] in ("method", "constructor"):
            if parent and parent["type"] == "class":
                participants[node_id] = {
                    "key": parent["node_id"],
                    "label": parent["name"],
                    "kind": "class",
                    "file": parent["file"],
                }
                continue
        participants[node_id] = {
            "key": node["file"],
            "label": (node["file"] or "?").rsplit("/", 1)[-1],
            "kind": "module",
            "file": node["file"],
        }
    return participants


def _flow_mermaid(
    participants: Dict[str, Dict[str, Any]],
    messages: List[Dict[str, Any]],
    truncated: bool,
    pruned: int,
) -> str:
    """Render the flow as a mermaid ``sequenceDiagram`` (same grammar as the
    static visualize.py layer — one export path from either surface)."""
    lane_order: List[str] = []
    lane_ids: Dict[str, str] = {}
    for node_id in sorted(participants, key=lambda n: participants[n]["key"]):
        key = participants[node_id]["key"]
        if key not in lane_ids:
            lane_ids[key] = f"P{len(lane_ids)}"
            lane_order.append(key)
    by_key: Dict[str, str] = {}
    for node_id, participant in participants.items():
        by_key[node_id] = lane_ids[participant["key"]]

    lines = ["sequenceDiagram", "    autonumber"]
    for key in lane_order:
        first_label = next(
            participants[n]["label"] for n in participants if participants[n]["key"] == key
        )
        lines.append(f'    participant {lane_ids[key]} as "{_mermaid_safe(first_label)}"')
    for msg in messages:
        src, dst = by_key.get(msg["from"]), by_key.get(msg["to"])
        if src is None or dst is None:
            continue
        suffix = " ⟲" if msg["revisit"] else ""
        line_note = f"  (L{msg['call_line']})" if msg["call_line"] else ""
        lines.append(f"    {src}->>{dst}: {_mermaid_safe(msg['label'])}{line_note}{suffix}")
    if truncated and lane_order:
        span = (
            lane_ids[lane_order[0]]
            if len(lane_order) == 1
            else f"{lane_ids[lane_order[0]]},{lane_ids[lane_order[-1]]}"
        )
        lines.append(
            f"    Note over {span}: truncated - {pruned} further message(s) pruned by caps"
        )
    return "\n".join(lines) + "\n"


# --- Class diagrams (ADR-033: type neighborhoods as mermaid classDiagram) --
#
# The flow view answers "how does a request move"; the class diagram answers
# "what is this type and what is it related to". Carrier edges: INHERITS
# (class -> class, both directions matter — bases above, subclasses below)
# and class-level CALLS/REFERENCES aggregation for uses links. Only the
# focal class gets its members (methods with signatures); relatives stay
# name-only so real hierarchies stay readable.

_CLASS_TYPES = ("class", "interface", "struct", "enum")
_CLASS_MAX_ANCESTRY = 4


def class_diagram(
    db_path,
    symbol: str,
    *,
    node_id: Optional[str] = None,
    max_members: int = 25,
    max_related: int = 8,
) -> Dict[str, Any]:
    """Type neighborhood for one class as a mermaid ``classDiagram``.

    The focal class carries its methods (signatures where the graph has
    them); bases and capped subclasses arrive via INHERITS, and uses
    associations via class-level CALLS/REFERENCES. Like ``flow``:
    ``node_id`` pins an ambiguous name, ``{"candidates": ...}``
    disambiguates, ``{"entry": None}`` means unknown. Related classes are
    returned as re-entry targets for the UI. Members are capped (god
    classes stay readable; the cap is surfaced as a mermaid note and in
    ``members_truncated``).
    """
    conn = _connect(db_path)
    try:
        entries = _class_entry_candidates(conn, symbol, node_id)
        if not entries:
            return {"entry": None}
        if len(entries) > 1:
            return {"candidates": entries}
        entry = entries[0]

        members = [
            {"name": row["name"], "signature": row["signature"]}
            for row in conn.execute(
                """
                SELECT name, signature FROM graph_node
                WHERE parent_id = ? AND type IN ("method", "constructor")
                ORDER BY name, node_id
                """,
                (entry["node_id"],),
            )
        ]
        bases = _class_relatives(conn, entry["node_id"], "out")
        subclasses = _class_relatives(conn, entry["node_id"], "in", max_related)
        associates = _class_associates(conn, entry["node_id"], max_related)
        related = _dedupe_related(entry, bases, subclasses, associates)
        members_total = len(members)
        members_truncated = members_total > max_members
        members = members[:max_members]

        return {
            "entry": {k: entry[k] for k in ("node_id", "name", "type", "file", "line")},
            "members": members,
            "members_truncated": members_truncated,
            "members_total": members_total,
            "bases": [dict(r) for r in bases],
            "subclasses": [dict(r) for r in subclasses],
            "associates": [dict(r) for r in associates],
            "related": related,
            "truncated": len(subclasses) >= max_related or len(associates) >= max_related,
            "mermaid": _class_mermaid(entry, members, bases, subclasses, associates, members_total),
        }
    finally:
        conn.close()


def _class_entry_candidates(
    conn: sqlite3.Connection, symbol: str, node_id: Optional[str]
) -> List[Dict[str, Any]]:
    placeholders = ", ".join("?" for _ in _CLASS_TYPES)
    if node_id:
        rows = conn.execute(
            f"""
            SELECT node_id, name, type, file, line FROM graph_node
            WHERE node_id = ? AND type IN ({placeholders})
            """,
            (node_id, *_CLASS_TYPES),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT node_id, name, type, file, line FROM graph_node
            WHERE name = ? AND type IN ({placeholders})
            ORDER BY file, line, node_id
            """,
            (symbol, *_CLASS_TYPES),
        ).fetchall()
    return [dict(row) for row in rows]


def _class_relatives(
    conn: sqlite3.Connection,
    node_id: str,
    direction: str,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """INHERITS neighbors, edges being subclass -> base (src is the child):
    ``out`` walks the focal class's bases (to ancestry depth), ``in``
    collects direct subclasses (capped)."""
    column, join = ("src", "dst") if direction == "out" else ("dst", "src")
    rows: List[Dict[str, Any]] = []
    frontier = [node_id]
    seen = {node_id}
    for _ in range(_CLASS_MAX_ANCESTRY if direction == "out" else 1):
        placeholders = ", ".join("?" for _ in frontier)
        found: Dict[str, Dict[str, Any]] = {}
        for row in conn.execute(
            f"""
            SELECT n.node_id, n.name, n.type, n.file, n.line
            FROM graph_edge e JOIN graph_node n ON n.node_id = e.{join}
            WHERE e.type = "INHERITS" AND e.{column} IN ({placeholders})
            ORDER BY n.name, n.node_id
            """,
            tuple(frontier),
        ):
            found[row["node_id"]] = dict(row)
        if not found:
            break
        for relative in found.values():
            if relative["node_id"] not in seen:
                seen.add(relative["node_id"])
                rows.append(relative)
        if direction == "in" or len(rows) >= limit:
            rows = rows[:limit]
            break
        frontier = list(found)
    return rows


def _class_associates(conn: sqlite3.Connection, node_id: str, limit: int) -> List[Dict[str, Any]]:
    """Classes the focal class's methods call or reference (uses/creates),
    aggregated to the target class, ranked by link count then name."""
    counts: Dict[str, int] = defaultdict(int)
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    for row in conn.execute(
        """
        SELECT e.type, d.node_id, d.name, d.type AS dtype, d.file, d.line
        FROM graph_edge e
        JOIN graph_node s ON s.node_id = e.src
        JOIN graph_node d ON d.node_id = e.dst
        WHERE e.type IN ("CALLS", "REFERENCES")
          AND s.parent_id = ?
          AND d.type IN ("class", "interface", "struct", "enum")
          AND d.node_id != ?
        """,
        (node_id, node_id),
    ):
        counts[row["node_id"]] += 1
        rows_by_id[row["node_id"]] = dict(row)
    ranked = sorted(counts, key=lambda nid: (-counts[nid], rows_by_id[nid]["name"], nid))[:limit]
    return [{**rows_by_id[nid], "links": counts[nid]} for nid in ranked]


def _dedupe_related(entry, bases, subclasses, associates) -> List[Dict[str, Any]]:
    """The flat re-entry list (UI jumps to another class's diagram)."""
    seen = {entry["node_id"]}
    related: List[Dict[str, Any]] = []
    for group, relation in ((bases, "base"), (subclasses, "subclass"), (associates, "associate")):
        for row in group:
            if row["node_id"] not in seen:
                seen.add(row["node_id"])
                related.append(
                    {
                        "node_id": row["node_id"],
                        "name": row["name"],
                        "file": row["file"],
                        "relation": relation,
                    }
                )
    related.sort(key=lambda r: (r["relation"], r["name"], r["node_id"]))
    return related


def _class_mermaid(entry, members, bases, subclasses, associates, members_total: int = 0) -> str:
    """Render the neighborhood as a mermaid ``classDiagram`` — same
    sanitize-everything rule as the flow view (names are untrusted text)."""
    lines = ["classDiagram"]

    def ident(row: Dict[str, Any]) -> str:
        # Mermaid class ids: no spaces; keep it readable and stable.
        return _mermaid_safe(row["name"]).replace(" ", "_")

    lines.append(f"    class {ident(entry)} {{")
    for member in members:
        text = member["signature"] or member["name"]
        # Drop the return-annotation tail; the diagram wants the call shape.
        call = text.split(" -> ")[0].strip()
        lines.append(f"        {_mermaid_safe(call)}")
    lines.append("    }")
    if members_total > len(members):
        lines.append(
            f'    note for {ident(entry)} "{members_total} members - showing first {len(members)}"'
        )
    for row in bases:
        lines.append(f"    {ident(row)} <|-- {ident(entry)}")
    for row in subclasses:
        lines.append(f"    {ident(entry)} <|-- {ident(row)}")
    for row in associates:
        lines.append(f"    {ident(entry)} --> {ident(row)} : uses")
    return "\n".join(lines) + "\n"


def ffi_surface(db_path) -> Dict[str, Any]:
    """The cross-language FFI callsite map: which python wrapper calls
    which rust function through the PyO3 boundary.

    Statement-level: CCG statement ``uses`` ∩ rust function names — the
    identifier sets the enrichment pass already carries. Returns
    ``{python_file: [rust_function, ...]}`` plus a per-function call
    count. The consumer surfaces this in the inspector (a class diagram
    associate list, a flow participant) — symbol-level REFERENCES edges
    across the language boundary need import-name persistence first.
    """
    conn = _connect(db_path)
    try:
        rust_fns = {
            row["name"]: {"file": row["file"], "line": row["line"]} for row in conn.execute("""
                SELECT name, file, line FROM graph_node
                WHERE file LIKE 'rust/crates/python-bindings/src/%' AND type = 'function'
                """)
        }
        if not rust_fns:
            return {"callsites": {}, "rust_functions": 0, "statements": 0}
        callsites: Dict[str, List[Dict[str, Any]]] = {}
        statement_count = 0
        for row in conn.execute("""
            SELECT name, file, line, metadata FROM graph_node
            WHERE statement_type IS NOT NULL
              AND metadata LIKE '%uses%'
              AND (file LIKE 'victor/native/%' OR file LIKE 'victor/processing/native/%')
            """):
            try:
                uses = set(json.loads(row["metadata"]).get("uses", []))
            except (ValueError, TypeError):
                continue
            hits = uses & set(rust_fns)
            if not hits:
                continue
            statement_count += 1
            entry = callsites.setdefault(row["file"], [])
            for fn_name in sorted(hits):
                target = rust_fns[fn_name]
                if not any(c["rust_function"] == fn_name for c in entry):
                    entry.append(
                        {
                            "rust_function": fn_name,
                            "rust_file": target["file"],
                            "rust_line": target["line"],
                            "callsite_line": row["line"],
                        }
                    )
        return {
            "callsites": callsites,
            "rust_functions": len(
                {c["rust_function"] for entries in callsites.values() for c in entries}
            ),
            "statements": statement_count,
        }
    finally:
        conn.close()


def deployment_diagram(db_path) -> Dict[str, Any]:
    """Victor's deployment topology as a mermaid deployment-style flow.

    Composes on the existing graph data: the package structure (what
    exists), the FFI/IPC boundaries (how the languages connect), and the
    integration surfaces (where the system meets the outside). Not a
    UML deploymentDiagram in the strict sense (no physical nodes) — the
    useful question here is "what are the runtime surfaces, what do they
    depend on, and where does the language boundary sit."
    """
    conn = _connect(db_path)
    try:
        # Count files per surface to prove presence in the graph
        surfaces: Dict[str, int] = {}
        for surface, pattern in (
            ("CLI / TUI", "victor/ui/%"),
            ("HTTP API", "victor/integrations/api/%"),
            ("MCP Server", "victor/integrations/mcp/%"),
            ("VS Code Extension", "vscode-victor/%"),
            ("Core Framework", "victor/framework/%"),
            ("Agent Runtime", "victor/agent/%"),
            ("Graph / Indexing", "victor/core/%"),
            ("Native Engine (Rust)", "rust/crates/%"),
            ("LLM Providers", "victor/providers/%"),
            ("Vertical Plugins", "verticals/%"),
            ("Codegraph SDK", "victor-codegraph/%"),
            ("Contracts SDK", "victor-contracts/%"),
        ):
            n = conn.execute(
                "SELECT COUNT(DISTINCT file) FROM graph_node WHERE file LIKE ? AND type != 'statement'",
                (pattern,),
            ).fetchone()[0]
            if n:
                surfaces[surface] = n

        # The dependency edges between surfaces (hard-coded from the
        # architecture; a future version could derive them from IMPORTS)
        topology = [
            ("CLI / TUI", "Core Framework"),
            ("CLI / TUI", "Agent Runtime"),
            ("HTTP API", "Core Framework"),
            ("HTTP API", "Agent Runtime"),
            ("VS Code Extension", "HTTP API"),
            ("MCP Server", "Core Framework"),
            ("Core Framework", "Agent Runtime"),
            ("Agent Runtime", "Graph / Indexing"),
            ("Agent Runtime", "LLM Providers"),
            ("Agent Runtime", "Vertical Plugins"),
            ("Graph / Indexing", "Codegraph SDK"),
            ("Graph / Indexing", "Native Engine (Rust)"),
            ("Graph / Indexing", "Contracts SDK"),
            ("Vertical Plugins", "Contracts SDK"),
            ("Core Framework", "Contracts SDK"),
        ]

        # Mermaid: flowchart LR (deployment diagrams in mermaid use the
        # flowchart syntax with subgraphs for grouping)
        lines = ["flowchart LR"]
        lines.append("    subgraph Client_Surfaces[Client Surfaces]")
        lines.append('        CLI["CLI / TUI"]')
        lines.append('        VS["VS Code Extension"]')
        lines.append('        MCP["MCP Server"]')
        lines.append("    end")
        lines.append("    subgraph Server[Server]")
        lines.append('        API["HTTP API"]')
        lines.append('        FW["Core Framework"]')
        lines.append('        AG["Agent Runtime"]')
        lines.append('        VT["Vertical Plugins"]')
        lines.append("    end")
        lines.append("    subgraph Data_Layer[Data & Intelligence]")
        lines.append('        GI["Graph / Indexing"]')
        lines.append('        CG["Codegraph SDK"]')
        lines.append('        CT["Contracts SDK"]')
        lines.append('        RUST["Native Engine (Rust)"]')
        lines.append('        LLM["LLM Providers"]')
        lines.append("    end")
        lines.append("    CLI --> FW")
        lines.append("    CLI --> AG")
        lines.append("    VS --> API")
        lines.append("    MCP --> FW")
        lines.append("    API --> FW")
        lines.append("    API --> AG")
        lines.append("    FW --> AG")
        lines.append("    FW --> CT")
        lines.append("    AG --> GI")
        lines.append("    AG --> LLM")
        lines.append("    AG --> VT")
        lines.append("    AG --> CT")
        lines.append("    GI --> CG")
        lines.append("    GI --> RUST")
        lines.append("    GI --> CT")
        lines.append("    VT --> CT")
        lines.append("    style RUST fill:#fef3c7,stroke:#d97706,stroke-dasharray: 5 5")
        lines.append("    style VS fill:#dbeafe,stroke:#2563eb")

        return {
            "surfaces": surfaces,
            "topology": topology,
            "mermaid": "\n".join(lines) + "\n",
        }
    finally:
        conn.close()
