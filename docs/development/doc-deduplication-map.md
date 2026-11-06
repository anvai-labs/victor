# Documentation Deduplication Map

**Date:** 2026-11-06  
**Status:** Active ledger — verification data + prioritized consolidation plan  
**Method:** heading-set similarity (normalized H2+ jaccard ≥ 0.25 or ≥ 5 shared) over all
364 tracked `.md`/`.adoc` files, with template/record families classified out and every
cluster verified by excerpt diff, git recency, nav membership and inbound links before
verdict. Re-run the inventory before acting on a cluster — files move.

## Inventory (364 files)

| Family | Count | Dedup policy |
|---|---|---|
| Narrative docs (`docs/**` minus records, READMEs, reference) | 243 | In scope |
| `docs/guides/` (unpublished legacy tree) | 38 | In scope (program) |
| FEP records (`feps/`) | 44 | Out of scope — shared skeleton is the template, by design |
| ADR records (`docs/architecture/adr/`) | 35 | Out of scope — same |
| Review records (`docs/reviews/**`) | ~30 | Out of scope — review template |
| Changelogs / CLAUDE.md / adoc book | 5 | See clusters below |

## Clusters and verdicts

### 1. Troubleshooting — DONE (2026-11-06, this PR)

`docs/guides/TROUBLESHOOTING.md` (981 lines, not in mkdocs nav, one inbound link)
duplicated `docs/user-guide/troubleshooting.md` (nav-published). Salvaged the unique
Provider Model Registration runbook into the canonical guide; deleted the legacy file;
removed the repository-map entry. Net −798 lines.

### 2. Providers guide — DONE (2026-11-06)

The recorded "16 unique H2s" dissolved under verification: per-provider sections
(anthropic/google/moonshot/xai) were *renamed and expanded* in the published doc
(Anthropic Claude, Google Gemini, Moonshot (Kimi), xAI Grok), and the capability
matrices carried stale model names (Claude 3.5/GPT-4/Gemini 2.0) duplicating the
published Tool Calling / Vision sections with worse data. The directory held two
more unpublished companions (`comparison.md` 12L, `setup.md` 83L), both superseded.

**Executed:** salvaged the verified-live Startup Behavior note (lazy registration +
`VICTOR_ENABLE_MLX_PROVIDER` gate, confirmed in `victor/providers/registry.py`);
deleted `docs/reference/providers/` (3 files); repointed all inbound links.
Lesson: heading-level "unique" counts overstate divergence — verify against renamed
equivalents and check data freshness before salvaging tables.

### 3. Configuration guide — DONE (2026-11-06)

The orphan's "40 unique H2s" had two explanations, both terminal: (a) its second half
is a *project-context template* (Project Overview / Tech Stack / Git Workflow /
Development Commands) superseded by `victor init` runtime synthesis
(`victor/framework/init_synthesizer.py`); (b) its yaml "depth" documents **dead
config keys** — `api_base_env`, `auto_compile`, `cache_compiled`, `compress_history`,
`allowed_resources` exist nowhere in the code, and `fallback_providers` is a
`ResilientProvider` SDK argument misattributed as a profiles key. Content-level token
coverage against the two published docs confirmed everything live is already documented.

**Executed:** deleted `docs/reference/configuration/index.md`; published the verified-
alive companion as `docs/reference/api-keys.md` (all its `victor keys` commands and
provider env vars exist in code; nav entry added); repointed all inbound links.
Lesson: verify "unique" keys against the codebase — fossil option names are the most
dangerous duplication, because they look like depth.

### 4. CLI reference — DONE (2026-11-06)

**Executed:** merged the six genuinely-absent commands (`dashboard`, `docs`,
`embeddings`, `examples`, `security`, `test-provider`) plus the unique Exit Codes
table into `docs/user-guide/cli-reference.md`; deleted `docs/reference/cli-commands.md`
and its nav entry; repointed links (`docs/index.md`, repository map,
`reference/quick-reference.md`). Initial "8 unique commands" included false positives
from heading-format differences — coverage checks must run against full text, not
heading equality.

### 5. `docs/guides/` legacy tree — P3 program (RECLASSIFIED 2026-11-06)

Post-execution rescan against published counterparts changes the verdict: most of
the tree (OBSERVABILITY, MCP_INTEGRATION, BENCHMARKING, EDGE_MODEL, WORKFLOW_*
HITL, MULTI_AGENT_*, TASK_PLANNER, development/, integration/, observability/)
has **no published counterpart at heading level** — it is *orphaned unique
content*, not duplication. P3 is therefore a publish-or-archive triage program
(per doc: is it current? where does it belong?), not a dedup pass. Genuine
remaining duplicates inside the tree: `task_completion.md` vs
`victor/tools/README_RESEARCH_TASK_TYPE.md` (0.28 overlap) and the three
pointer stubs (keep).

### 6. No action (verified non-duplicates)

- **Vertical docs** (`docs/verticals/*.md`, 5 files, 11-14 shared H2s) and **vertical
  READMEs** (`verticals/*/README.md`): shared *template* structure, distinct content
  per vertical. Intentional.
- **`VICTOR_AGENT_GUIDE.adoc`** (766 lines): standalone AsciiDoc book/slide artifact
  (HTML/PDF with diagrams), referenced from CHANGELOG and the repository map. Keep —
  but note version drift: it self-identifies v0.7.5 against a 0.26.x codebase; refresh
  on the next book pass.
- **FEP/ADR/review records**: template skeletons are the format, not duplication.
- **`docs/user-guide/index.md` vs topic docs**: index summary links, normal.

## Execution rules (from the completed clusters)

1. Never delete on heading-similarity alone — verify by excerpt diff, git recency,
   nav membership and inbound links (records have repeatedly diverged 50%+).
2. Salvage unique sections before deleting; the loser often holds real content.
3. Fix inbound links in the same change; `make check-repo-hygiene` must pass.
4. Follow the remove-during-consolidation precedent (roadmap/ARCHITECTURE stubs),
   but keep deliberate canonical-pointer stubs.
5. Update this ledger and `docs/development/repository-map.md` when trees change.
