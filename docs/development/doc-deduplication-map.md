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

### 2. Providers guide — P1 merge

| File | Lines | Nav | Last touched |
|---|---|---|---|
| `docs/user-guide/providers.md` | 957 | yes | 2026-10-11 |
| `docs/reference/providers/index.md` | 1026 | **no (orphan fork)** | 2026-09-08 |

23 shared H2s with verbatim-identical sections ("Choosing a Provider / Decision Guide").
The orphan fork additionally carries 16 unique H2s (per-provider sections: anthropic,
google, moonshot, ollama; feature-support and comparison matrices). ~5 inbound links
from `docs/user-guide/*.md` point at the orphan.

**Action:** merge the fork's unique per-provider sections and matrices into the
published doc; repoint inbound links; delete the orphan.

### 3. Configuration guide — P1 resolve (not a blind delete)

| File | Lines | Nav | Last touched |
|---|---|---|---|
| `docs/getting-started/configuration.md` | 713 | yes | 2026-09-08 |
| `docs/reference/configuration/index.md` | 908 | **no (orphan)** | 2026-09-08 |
| `docs/reference/configuration-options.md` | 643 | yes | 2026-10-28 |

28 shared H2s between the first two; the orphan has 40/55 unique H2s (priority order,
directory structure, numbered best practices) — it is a deeper *guide*, not a stale
copy. `configuration-options.md` shares **zero** headings with the orphan: it is the
per-option reference table, complementary to both.

**Action:** de-duplicate the intro-level overlap into `getting-started/configuration.md`,
publish the orphan's deep content under the reference section (or fold it in), and keep
`configuration-options.md` as the option table. Decision needed on placement; content
merge is mechanical after that.

### 4. CLI reference — P2 merge

| File | Lines | Nav | Last touched |
|---|---|---|---|
| `docs/user-guide/cli-reference.md` | 746 | yes | 2026-09-08 |
| `docs/reference/cli-commands.md` | 701 | **yes (both published!)** | 2026-06-16 |

Overlapping command coverage in different formats; the stale June doc still documents
commands the September doc lacks (dashboard, docs, embeddings, examples, index,
security).

**Action:** merge unique command entries into `cli-reference.md`; remove the
`reference/cli-commands.md` nav entry and file; fix inbound links.

### 5. `docs/guides/` legacy tree — P3 program

38 unpublished docs parallel to `getting-started/` and `user-guide/`. The three
canonical-pointer stubs (`installation.md`, `quickstart.md`, `first-agent.md`, 6-10
lines) are routing — keep. The remaining ~30 need the cluster-1 treatment one category
at a time (workflow guides, MCP, observability, multi-agent...). Not urgent: they are
invisible to site users, but they are drift bait — every future edit risks forking
published content again.

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
