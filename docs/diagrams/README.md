# Diagrams

**Canonical diagrams live inline** in the master documents, where they are rendered by
GitHub/MkDocs and kept in sync with the prose:

| Master doc | Diagrams |
|------------|----------|
| [`architecture.md`](../architecture.md) | System overview, layers, services, providers, tools, workflows, teams, state, DB, config, extensions |
| [`features.md`](../features.md) | Feature mindmap, capability catalog |
| [`tech-stack.md`](../tech-stack.md) | Dependency map, ER diagram, tech-debt gantt |
| [`roadmap.md`](../roadmap.md) | Execution timeline, directional horizons |

This directory holds the **editable Mermaid sources** (`.mmd`) for diagrams that are also
rendered to `.svg`. The `.svg` files are build artifacts — regenerate them from the
`.mmd` sources; do not hand-edit the `.svg`.

## Sources

- **`architecture/`** — high-level architecture diagrams. `victor_0_7_architecture.mmd` is
  the canonical 0.7 layered map; `victor_0_7_readme_architecture.svg` is the variant used by
  the root `README.md`. `system-overview`, `provider-system`, `config-system`, and
  `multi-agent` are focused component diagrams referenced from `architecture.md`.
- **`sequences/`** — sequence diagrams for key flows: `tool-execution`, `provider-switch`,
  `workflow-execution` (each has a rendered `.svg`).
- **`victor-guide/`** — the 12 diagrams (`d01`–`d12`) for the
  [`VICTOR_AGENT_GUIDE.adoc`](https://github.com/anvai-labs/victor/blob/develop/VICTOR_AGENT_GUIDE.adoc) presentation deck.

## Re-rendering

```bash
# One diagram
mmdc -i architecture/system-overview.mmd -o architecture/system-overview.svg -b white -w 1400

# All sources
find docs/diagrams -name '*.mmd' -exec sh -c \
  'mmdc -i "$0" -o "${0%.mmd}.svg" -b white -w 1400' {} \;
```

For the `victor-guide/` deck, SVG is used for HTML and PNG (at 2×) for the PDF backend —
see the render runbook in `VICTOR_AGENT_GUIDE.adoc`.
