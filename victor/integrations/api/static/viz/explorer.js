/* Victor Code Explorer (ADR-033 phase 1): package -> module -> class
   navigation as zoom states of one Cytoscape canvas. */
"use strict";

const cyContainer = document.getElementById("cy");
const inspectorBody = document.getElementById("inspector-body");
const state = {
  level: "package",
  package: null,
  module: null,
  history: [],
  flowNodeId: null,
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
    cy.json({ elements: els });
    cy.layout({
      name: "concentric",
      // Degree spans 0..~87k on this repo — raw values blow the concentric
      // rings far off-screen; log-scale keeps the ordering readable.
      concentric: (n) => Math.log10(1 + (n.data("degree") || 0)),
      levelWidth: () => 2,
    }).run();
    showHint("Click a package to zoom into its modules.");
  } catch (e) { showHint(`Error: ${e.message}`); }
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

async function loadClass(module) {
  hideFlow();
  state.level = "class"; state.module = module;
  updateNav();
  try {
    const els = await fetchElements(`/api/graph/class?module=${encodeURIComponent(module)}`);
    cy.json({ elements: els });
    cy.layout({ name: "dagre", rankDir: "TB", nodeSep: 30 }).run();
    showHint(`Symbols in ${module}.`);
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

document.getElementById("btn-package").addEventListener("click", () => loadPackage());
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
