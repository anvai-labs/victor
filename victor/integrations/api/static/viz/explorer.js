/* Victor Code Explorer (ADR-033 phase 1): package -> module -> class
   navigation as zoom states of one Cytoscape canvas. */
"use strict";

const cyContainer = document.getElementById("cy");
const inspectorBody = document.getElementById("inspector-body");
const matrixView = document.getElementById("matrix-view");
const matrixTable = document.getElementById("matrix-table");

const state = {
  level: "package",
  package: null,
  module: null,
  history: [],
  flowNodeId: null,
  matrixMode: false,
};

const cy = cytoscape({
  container: cyContainer,
  elements: [],
  style: [
    { selector: "node", style: {
      label: "data(label)", "text-valign": "center", "font-size": "10px",
      "background-color": "#e0e7ff", color: "#111", width: "label", height: 20,
      "padding-left": "6px", "padding-right": "6px", "border-width": 1, "border-color": "#94a3b8",
    }},
    { selector: 'node[kind = "package"]', style: { "background-color": "#bfdbfe", "font-weight": "bold" } },
    { selector: "edge", style: {
      width: "mapData(weight, 1, 500, 1, 6)", "line-color": "#94a3b8",
      "target-arrow-shape": "triangle", "arrow-scale": 0.6, "curve-style": "bezier",
    }},
    { selector: "node:selected", style: { "border-color": "#2563eb", "border-width": 3 } },
  ],
  layout: { name: "grid" },
});

function esc(text) {
  // Escape untrusted text at the DOM boundary (symbol names, files, errors).
  const el = document.createElement("span");
  el.textContent = String(text);
  return el.innerHTML;
}

async function fetchElements(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

function inspect(node) {
  const d = node.data();
  const fields = Object.entries(d)
    .filter(([k]) => !["id", "label"].includes(k))
    .map(([k, v]) => `<div class="field"><strong>${esc(k)}</strong>${esc(v)}</div>`)
    .join("");
  inspectorBody.innerHTML = fields || '<p class="hint">No metadata.</p>';
  const flowButton = (label, type) => {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = "flow-action";
    btn.addEventListener("click", () => {
      pushHistory();
      showFlow(d.label, d.id, type);
    });
    return btn;
  };
  if (d.kind === "class" || d.kind === "interface" || d.kind === "struct" || d.kind === "enum") {
    inspectorBody.appendChild(flowButton("Show class diagram", "class"));
  } else if (["function", "method", "constructor"].includes(d.kind)) {
    inspectorBody.appendChild(flowButton("Trace call flow", "flow"));
  }
}

function showHint(text) {
  inspectorBody.innerHTML = `<p class="hint">${text}</p>`;
}

async function loadPackage(top = 12) {
  hideFlow();
  state.level = "package"; state.package = null; state.module = null;
  updateNav();
  try {
    const els = await fetchElements(`/api/graph/package?top=${top}`);
    if (state.matrixMode) {
      renderCouplingMatrix(els);
      return;
    }
    groupComponents(els, { minSize: 2, label: (c) => `coupled group (${c.length})` });
    cy.json({ elements: els });
    cy.layout({
      name: "concentric",
      // Degree spans 0..~87k on this repo — raw values blow the concentric
      // rings far off-screen; log-scale keeps the ordering readable.
      concentric: (n) => Math.log10(1 + (n.data("degree") || 0)),
      levelWidth: () => 2,
    }).run();
    const groups = els.nodes.filter((n) => n.data.parent).length
      ? new Set(els.nodes.map((n) => n.data.parent).filter(Boolean)).size
      : 0;
    showHint(
      `${els.nodes.filter((n) => !n.data.parent).length} packages` +
        (groups ? ` in ${groups} dashed coupling groups — nodes inside an envelope are coupled; bare nodes have no cross-package edges in this top-set.` : " (no coupling edges in this top-set).") +
        " Click a package to zoom into its modules."
    );
  } catch (e) { showHint(`Error: ${e.message}`); }
}

function groupComponents(els, { minSize = 2, label } = {}) {
  // Connected components become cytoscape COMPOUND parents: the renderer
  // draws each parent as a live envelope around its children, so coupled
  // groups travel together and read as one island. Parents are prepended
  // (they must exist before children reference them).
  const parents = [];
  let island = 0;
  for (const group of componentGroups(els)) {
    if (group.length < minSize) continue;
    island += 1;
    const id = `island-${island}`;
    for (const n of group) n.data.parent = id;
    parents.push({
      data: { id, label: label ? label(group) : `group ${island}`, kind: "island" },
    });
  }
  els.nodes = [...parents, ...els.nodes];
}

async function loadModule(pkg) {
  hideFlow();
  // "package" is a strict-mode reserved word — it cannot be a binding name.
  state.level = "module"; state.package = pkg; state.module = null;
  updateNav();
  try {
    const els = await fetchElements(`/api/graph/module/${encodeURIComponent(pkg)}`);
    cy.json({ elements: els });
    cy.layout({ name: "dagre", rankDir: "TB", nodeSep: 40 }).run();
    showHint(`Modules of ${pkg}. Click a module for its symbols.`);
  } catch (e) { showHint(`Error: ${e.message}`); }
}

function renderCouplingMatrix(els) {
  // Packages as a row/column heat matrix: every pair comparable, no edge
  // crossings; intensity = log-scaled coupling weight. Header clicks drill
  // into that package's module view; cells title their exact weight.
  const pkgs = els.nodes
    .map((n) => n.data)
    .sort((a, b) => (b.degree || 0) - (a.degree || 0));
  const weight = new Map();
  let max = 1;
  for (const e of els.edges) {
    const d = e.data;
    weight.set(`${d.source}\u0000${d.target}`, d.weight || 1);
    weight.set(`${d.target}\u0000${d.source}`, d.weight || 1);
    max = Math.max(max, d.weight || 1);
  }
  const short = (label) => label.split("/").pop();
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th"));
  for (const p of pkgs) {
    const th = document.createElement("th");
    th.className = "pkg-top";
    th.textContent = short(p.label);
    th.title = `${p.label} (degree ${p.degree || 0})`;
    th.addEventListener("click", () => { state.matrixMode = false; setMatrixMode(false); loadModule(p.id); });
    head.appendChild(th);
  }
  matrixTable.innerHTML = "";
  matrixTable.appendChild(head);
  for (const rowP of pkgs) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = short(rowP.label);
    th.title = `${rowP.label} (degree ${rowP.degree || 0})`;
    th.addEventListener("click", () => { setMatrixMode(false); loadModule(rowP.id); });
    tr.appendChild(th);
    for (const colP of pkgs) {
      const td = document.createElement("td");
      if (rowP.id === colP.id) {
        td.className = "diag";
      } else {
        const w = weight.get(`${rowP.id}\u0000${colP.id}`) || 0;
        td.className = "cell";
        if (w) {
          const t = Math.log10(1 + w) / Math.log10(1 + max);
          td.style.backgroundColor = `rgba(37, 99, 235, ${(0.15 + 0.75 * t).toFixed(2)})`;
          td.style.color = t > 0.55 ? "#fff" : "#475569";
          td.title = `${rowP.label} ↔ ${colP.label}: ${w} edges`;
        }
      }
      tr.appendChild(td);
    }
    matrixTable.appendChild(tr);
  }
  showHint("Coupling matrix — darker cells couple more (log scale). Click a header to open that package.");
}

function componentGroups(els) {
  // Union-find over the element edges -> the connected components (the
  // module-cohesion signal AND the packages-view island grouping). One
  // implementation: componentStats summarizes it, groupComponents wraps
  // it as cytoscape compound parents.
  const parent = new Map();
  const find = (x) => {
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root);
    return root;
  };
  const union = (a, b) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };
  for (const n of els.nodes) parent.set(n.data.id, n.data.id);
  for (const e of els.edges) union(e.data.source, e.data.target);
  const members = new Map();
  for (const n of els.nodes) {
    const r = find(n.data.id);
    if (!members.has(r)) members.set(r, []);
    members.get(r).push(n);
  }
  return [...members.values()];
}

function componentStats(els) {
  const groups = componentGroups(els);
  return { groups: groups.length, largest: Math.max(0, ...groups.map((g) => g.length)) };
}

async function loadClass(module) {
  hideFlow();
  state.level = "class"; state.module = module;
  updateNav();
  try {
    const els = await fetchElements(`/api/graph/class?module=${encodeURIComponent(module)}`);
    cy.json({ elements: els });
    // Edge COUNT was the wrong predicate (52 edges can concentrate on a
    // few hubs while 30+ symbols stay isolated — dagre then lays the
    // disconnected components side-by-side: a 3500x280px strip at ~0.13
    // zoom, effectively blank). CONNECTIVITY decides: grid the symbol
    // sheet when most symbols sit outside the largest linked group; dagre
    // when edges actually structure the module. The same component count
    // is the cohesion readout — sibling methods of ONE class are normally
    // unlinked (they collaborate through state); multiple UNLINKED
    // top-level symbols are a real co-location smell.
    const stats = componentStats(els);
    if (stats.largest * 2 < els.nodes.length) {
      const cols = Math.max(1, Math.ceil(Math.sqrt(els.nodes.length)));
      cy.layout({ name: "grid", cols, avoidOverlap: true, condense: true }).run();
      showHint(`${els.nodes.length} symbols · ${stats.groups} unlinked groups (largest ${stats.largest}) — sibling methods of one class are normally unlinked; multiple unlinked top-level symbols suggest the module co-locates unrelated jobs.`);
    } else {
      cy.layout({ name: "dagre", rankDir: "TB", nodeSep: 30 }).run();
      showHint(`${els.nodes.length} symbols · linked by ${els.edges.length} intra-module edges.`);
    }
    // (cohesion readout set above — this used to overwrite it)
  } catch (e) { showHint(`Error: ${e.message}`); }
}

let flowRenderSeq = 0;

function candidateRow(c) {
  const el = document.createElement("div");
  el.className = "field result-link";
  el.setAttribute("role", "button");
  el.setAttribute("tabindex", "0");
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); }
  });
  const where = c.line == null ? esc(c.file) : `${esc(c.file)}:${esc(c.line)}`;
  el.innerHTML = `<strong>${esc(c.type)} · ${where}</strong>${esc(c.name)}`;
  el.addEventListener("click", () => {
    state.flowNodeId = c.node_id;
    document.getElementById("flow-symbol").value = c.name;
    renderFlow();
  });
  return el;
}

async function showFlow(symbol, nodeId = null, type = null) {
  document.body.classList.add("flow-mode");
  document.getElementById("flow-view").hidden = false;
  cyContainer.style.display = "none";
  state.level = "flow";
  state.flowNodeId = nodeId;
  updateNav();
  document.getElementById("flow-symbol").value = symbol || "";
  if (type) document.getElementById("flow-type").value = type;
  if (symbol) await renderFlow();
}

function setMatrixMode(on) {
  state.matrixMode = on;
  matrixView.hidden = !on;
  cyContainer.style.display = on ? "none" : "";
  document.getElementById("btn-matrix").classList.toggle("active", on);
  if (state.level === "package") loadPackage();
}

function hideFlow() {
  document.body.classList.remove("flow-mode");
  document.getElementById("flow-view").hidden = true;
  cyContainer.style.display = "";
}

async function renderFlow() {
  const symbol = document.getElementById("flow-symbol").value.trim();
  const diagram = document.getElementById("flow-diagram");
  const badges = document.getElementById("flow-badges");
  const candidates = document.getElementById("flow-candidates");
  badges.textContent = "";
  candidates.innerHTML = "";
  if (!symbol) return;
  diagram.innerHTML = '<p class="hint">Tracing call flow…</p>';
  const type = document.getElementById("flow-type").value;
  const params = new URLSearchParams({ symbol });
  if (type === "flow") {
    params.set("depth", document.getElementById("flow-depth").value);
    params.set("granularity", document.getElementById("flow-granularity").value);
  }
  if (state.flowNodeId) params.set("node_id", state.flowNodeId);
  try {
    const res = await fetch(`/api/graph/${type === "flow" ? "flow" : "classdiagram"}?${params}`);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    if (body.candidates) {
      diagram.innerHTML = '<p class="hint">Multiple symbols share that name — pick one:</p>';
      body.candidates.forEach((c) => {
        candidates.appendChild(candidateRow(c));
      });
      return;
    }
    if (body.related) {
      // Class diagrams: related types are one-click re-entries.
      body.related.forEach((r) => {
        const el = candidateRow({ ...r, type: r.relation });
        candidates.appendChild(el);
      });
    }
    const flags = [];
    if (body.ordered === false) flags.push("unordered — reindex for call-line order");
    if (body.pruned) flags.push(`truncated (${body.pruned} pruned)`);
    if (body.members_truncated) flags.push(`${body.members_total} members — showing ${body.members.length}`);
    badges.textContent = flags.join(" · ");
    if (window.mermaid) {
      window.mermaid.initialize({ startOnLoad: false }); // strict: graph labels are untrusted
      const { svg } = await window.mermaid.render(`flow-svg-${++flowRenderSeq}`, body.mermaid);
      diagram.innerHTML = svg;
    } else {
      diagram.innerHTML = '<p class="hint">mermaid.min.js unavailable — showing source.</p>' +
        `<pre>${esc(body.mermaid)}</pre>`;
    }
  } catch (e) {
    diagram.innerHTML = `<p class="hint">Error: ${esc(e.message)}</p>`;
  }
}

function updateNav() {
  document.getElementById("btn-package").classList.toggle("active", state.level === "package");
  document.getElementById("btn-module").disabled = !state.package;
  document.getElementById("btn-module").classList.toggle("active", state.level === "module");
  document.getElementById("btn-class").disabled = !state.module;
  document.getElementById("btn-class").classList.toggle("active", state.level === "class");
  document.getElementById("btn-flow").classList.toggle("active", state.level === "flow");
  document.getElementById("btn-back").disabled = state.history.length === 0;
}

function pushHistory() {
  const snap = { level: state.level, package: state.package, module: state.module };
  state.history.push(snap);
}

cy.on("tap", "node", (evt) => {
  const node = evt.target;
  inspect(node);
  const d = node.data();
  if (d.kind === "package") { pushHistory(); loadModule(d.id); }
  else if (d.kind === "module") { pushHistory(); loadClass(d.id); }
});

document.getElementById("btn-package").addEventListener("click", () => { setMatrixMode(false); loadPackage(); });
document.getElementById("btn-matrix").addEventListener("click", () => setMatrixMode(!state.matrixMode));
document.getElementById("btn-module").addEventListener("click", () => state.package && loadModule(state.package));
document.getElementById("btn-class").addEventListener("click", () => state.module && loadClass(state.module));
document.getElementById("btn-flow").addEventListener("click", () => {
  pushHistory();
  showFlow(document.getElementById("flow-symbol").value.trim());
});
document.getElementById("flow-form").addEventListener("submit", (e) => {
  e.preventDefault();
  state.flowNodeId = null; // a typed name is a fresh lookup
  renderFlow();
});
document.getElementById("btn-back").addEventListener("click", () => {
  const prev = state.history.pop();
  if (!prev) return;
  if (prev.level === "package") { hideFlow(); loadPackage(); }
  else if (prev.level === "module") { hideFlow(); loadModule(prev.package); }
  else if (prev.level === "class") { hideFlow(); loadClass(prev.module); }
  else showFlow(document.getElementById("flow-symbol").value.trim(), state.flowNodeId);
});

document.getElementById("search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = document.getElementById("search-input").value.trim();
  if (!q) return;
  try {
    const res = await fetch(`/api/graph/search?q=${encodeURIComponent(q)}`);
    const body = await res.json();
    const items = (body.results || []).map((r) => {
      const d = r.data;
      return `<div class="field result-link" role="button" tabindex="0" data-module="${esc(d.file)}">` +
        `<strong>${esc(d.kind)} · ${esc(d.file)}:${esc(d.line)}</strong>${esc(d.label)}</div>`;
    }).join("");
    inspectorBody.innerHTML = items || '<p class="hint">No symbols matched.</p>';
    inspectorBody.querySelectorAll(".result-link").forEach((el) => {
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); }
      });
      el.addEventListener("click", () => {
        const mod = el.dataset.module;
        const pkg = mod.split("/").slice(0, 2).join("/");
        pushHistory();
        state.package = pkg;
        loadClass(mod);
      });
    });
  } catch (err) { showHint(`Search error: ${err.message}`); }
});

loadPackage();
