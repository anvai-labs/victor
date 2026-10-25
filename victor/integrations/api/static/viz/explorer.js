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
    .map(([k, v]) => `<div class="field"><strong>${k}</strong>${String(v)}</div>`)
    .join("");
  inspectorBody.innerHTML = fields || '<p class="hint">No metadata.</p>';
}

function showHint(text) {
  inspectorBody.innerHTML = `<p class="hint">${text}</p>`;
}

async function loadPackage(top = 12) {
  state.level = "package"; state.package = null; state.module = null;
  updateNav();
  try {
    const els = await fetchElements(`/api/graph/package?top=${top}`);
    cy.json({ elements: els });
    cy.layout({ name: "concentric", concentric: (n) => n.data("degree") || 0, levelWidth: 2 }).run();
    showHint("Click a package to zoom into its modules.");
  } catch (e) { showHint(`Error: ${e.message}`); }
}

async function loadModule(package) {
  state.level = "module"; state.package = package; state.module = null;
  updateNav();
  try {
    const els = await fetchElements(`/api/graph/module/${encodeURIComponent(package)}`);
    cy.json({ elements: els });
    cy.layout({ name: "dagre", rankDir: "TB", nodeSep: 40 }).run();
    showHint(`Modules of ${package}. Click a module for its symbols.`);
  } catch (e) { showHint(`Error: ${e.message}`); }
}

async function loadClass(module) {
  state.level = "class"; state.module = module;
  updateNav();
  try {
    const els = await fetchElements(`/api/graph/class?module=${encodeURIComponent(module)}`);
    cy.json({ elements: els });
    cy.layout({ name: "dagre", rankDir: "TB", nodeSep: 30 }).run();
    showHint(`Symbols in ${module}.`);
  } catch (e) { showHint(`Error: ${e.message}`); }
}

function updateNav() {
  document.getElementById("btn-package").classList.toggle("active", state.level === "package");
  document.getElementById("btn-module").disabled = !state.package;
  document.getElementById("btn-module").classList.toggle("active", state.level === "module");
  document.getElementById("btn-class").disabled = !state.module;
  document.getElementById("btn-class").classList.toggle("active", state.level === "class");
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
document.getElementById("btn-back").addEventListener("click", () => {
  const prev = state.history.pop();
  if (!prev) return;
  if (prev.level === "package") loadPackage();
  else if (prev.level === "module") loadModule(prev.package);
  else loadClass(prev.module);
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
      return `<div class="field result-link" data-module="${d.file}">` +
        `<strong>${d.kind} · ${d.file}:${d.line}</strong>${d.label}</div>`;
    }).join("");
    inspectorBody.innerHTML = items || '<p class="hint">No symbols matched.</p>';
    inspectorBody.querySelectorAll(".result-link").forEach((el) => {
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
