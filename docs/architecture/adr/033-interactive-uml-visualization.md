# ADR-033: Interactive multi-level code visualization (UML roadmap)

**Status:** Accepted (direction) — phase 1 (package→class explorer)
shipped #1502/#1503; phase 2 (call-flow sequence diagrams) shipped #1504
with call-site-line persistence on CALLS edges; phase 2b (class diagrams
as mermaid ``classDiagram`` type neighborhoods: INHERITS bases/subclasses,
uses associations, capped members with reported truncation) shipped on
the same panel — the Diagram panel is now family-parameterized.

**Date:** 2026-10-24

## Context

The tier-3-indexed graph (1.5M nodes / 3.9M edges on this repo) powers
retrieval, but its human-facing surface is thin: `victor index graph viz`
renders static mermaid snapshots at three fixed levels (package / module /
class), each an isolated view. Understanding a mechanism — how packages
collaborate, which classes inside them interact, how a request flows —
requires mentally stitching those snapshots.

## Decision (direction, not commitment)

Extend the codegraph visualization surface toward an **interactive,
multi-scale explorer** served over localhost, backed by the same
graph-query functions (`victor/core/graph_rag/visualize.py` stays the
query layer; the delivery changes):

1. **Unified package→class navigation.** One continuous view: scroll from
   package to package and see the classes within each interacting; select
   2–3 packages for a focused cross-package picture. Levels are zoom
   states of one graph, not separate diagrams.
2. **UML diagram families at the appropriate abstraction** (mermaid and/or
   C4 as the rendering grammar, chosen per diagram):
   - sequence diagrams: package- or class-level interaction traces (the
     graph's CALLS/REFERENCES chains + tier-3 compiler edges are the data)
   - class diagrams: per-module type relationships (INHERITS/IMPLEMENTS/
     COMPOSITION edges already exist)
   - use-case, activity, package, deployment diagrams: where the graph's
     data supports them (workflows/teams state for activity; vertical
     structure for package/deployment)
3. **Delivery: a local web app** (React + D3.js/cytoscape.js — decide by
   benchmark in design; D3 for custom layouts, cytoscape when the graph
   layouts suffice) served over `localhost`/`0.0.0.0` by a
   `victor viz serve` command; the graph queries stay Python-side behind
   a small HTTP surface. No cloud dependency; the project DB is the
   source.

## Consequences

- The `viz` subcommand family grows: `victor index graph viz` (static,
  keep — CI/scriptable) + `victor viz serve` (interactive, new).
- Phase 1 should be the unified package→class explorer — it reuses the
  existing query layer directly and delivers the "comprehensive
  mechanism" view the static levels cannot.
- UML families land incrementally AFTER the explorer, each driven by an
  actual understanding need (e.g. sequence diagrams from the tier-3 call
  graph), not by diagram-taxonomy completeness.
- Phase 2 (sequence diagrams) reads the ordering key from CALLS-edge
  ``call_line`` metadata (1-based first-use site from
  victor-codegraph's ``CodeRelation.call_site``, persisted since phase 2 —
  graphs indexed before it report ``ordered: false`` and fall back to a
  deterministic name order, never a guessed one). Participants: owning
  class for methods/constructors (constructor-call targets ARE lanes),
  module for free functions; caps and truncation are reported, never
  silent. Rendering: vendored mermaid UMD — the same grammar as the static
  layer, and the renderer for the remaining families (class, use-case,
  activity, deployment) without new vendored surface.
- Frontend assets are a new dependency surface for victor — vendored,
  no runtime CDN fetches (offline-first).

## Prior art in-repo

- `visualize.py` (level queries, deterministic output — the data layer)
- The d06–d13 deck diagrams (mermaid as a rendering grammar precedent)
- `vscode-victor` (an extension surface that could host the explorer
  later; start with the web app for iteration speed)
