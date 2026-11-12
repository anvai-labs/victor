# SQLite graph deduplication

**Date:** 2026-11-01
**Scope:** Phase 1 of `.victor/sandbox/codegraph-deduplication-handoff.md`.

## Handoff review

The handoff mixes source refactoring opportunities, legitimate inheritance and
storage duplication. A class with many incoming edges is not a duplicate. Two
`VerticalBase` definitions in the contracts and runtime layers describe distinct
source definitions. Matching names or signatures across files must not cause
their identities or relationships to collapse. Inheritance compression and
automatic cross-file aliases are outside this repair's scope.

The current persisted graph uses `graph_node` and `graph_edge`, not the proposed
`symbols`/`relationships` schema. The legacy `symbols` table has separate
consumers; overlapping records in two representations do not authorize dropping
either table. The existing graph indexer already tracks file content hashes and
embedding input, source and model-profile evidence. Phases 2–4 need a separate
gap analysis against that implementation.

The engine belongs to `victor/core/graph_rag/dedup.py`, alongside the existing
indexing pipeline. No new `victor/codegraph` package or native extension is needed.

## Definition identity and apply behavior

Discovery requires identical file, type, name, positive source range, signature,
parent, scope, language, documentation and other stored definition fields.
Source metadata, including qualified names and compiler IDs when present, must
agree. Embedding evidence may differ because it will be regenerated. Malformed
metadata, excluded embeddings and incomplete source locations are skipped.
This intentionally leaves ambiguous candidates for review. Duplicated parents
may hide child candidates until a later preview after the parent repair.

The lexicographically first ID is canonical. Apply validates fingerprints under
`BEGIN IMMEDIATE`, rejects caller-owned transactions and disabled foreign keys,
and commits all selected groups together. It rewrites both edge endpoints and
direct parent/scope references. Colliding edges keep the maximum non-null
weight; their original endpoints, weights, file lineage and metadata are retained
in `deduplication_edges` provenance. Original node identities and embedding
evidence are retained in `deduplication_sources`. FTS synchronization uses the
existing SQLite triggers.

Serialized subgraph/query caches and current module metrics are invalidated.
Historical metrics remain historical. Populated CPG fragment tables and unknown
edge schemas are rejected because this engine cannot reconcile them safely.
SQLite constraint/trigger failures roll back the entire repair. The command
shares the indexer's cross-process project lock; applications that bypass that
lock must be stopped during maintenance. Existing process-local caches are not
remotely invalidated.

External vectors cannot participate in the SQLite transaction. Each canonical
node loses its stale embedding evidence and receives a durable
`deduplication_pending` marker in the same commit as the merge. The command then
deletes obsolete vector IDs and regenerates only pending canonical embeddings
through the existing model/profile-validated indexing pipeline. It clears the
marker only after checking successful persisted embedding metadata. A failure
returns a nonzero status and retains the pending work for the next apply; it does
not claim to roll back committed graph rows or replay their merge. SQLite worker
writes settle before cancellation releases the project lock.
Vector IDs and legacy `symbol:`/`emb:` references still used by surviving graph
rows are protected from cleanup.

## Operation

From an environment containing this checkout and its current codegraph package:

```bash
victor index graph dedup --path /path/to/project --dry-run
victor index graph dedup --path /path/to/project --apply
```

`victor codegraph dedup` is a compatibility alias for the handoff's command.
Omitting both action flags previews; supplying both is an error. Preview opens
an existing database read-only and creates no backup or index lock. Only SQLite
graphs are supported; a non-SQLite project backend marker is rejected.

Every nonempty apply first creates a mode-0600 backup beside the database using
SQLite's backup API, including committed WAL contents. The backup is named
`project.db.dedup-<timestamp>-<unique-id>.bak`. Backup failure stops the operation.
Keep it until graph queries and embedding search have been checked. If embedding
refresh fails, rerun the same apply command after restoring model/vector-store
availability. Recovery of a database backup requires stopping writers and
restoring it with SQLite-aware tooling; the backup does not restore external
vector-store state.

## Measured discovery evidence

A read-only snapshot taken during this session contained 510,435 nodes and
1,046,478 edges, with **zero** eligible duplicate groups. The live project index
changed during the session, so these are snapshot counts. Five discovery scans
measured 113.323 ms median wall time, 109.038 ms median CPU time, and 1,588.228 ms
maximum wall time including the first scan. Traced Python allocation peaked at
3,682 bytes during discovery; whole-process peak RSS was 51,331,072 bytes on
macOS. The profile attributed essentially all scan time to SQLite execution.

Reproduce with:

```bash
python -m scripts.benchmark_graph_dedup /path/to/project/.victor/project.db --runs 5
```

No live graph merges were needed or applied. The handoff's 15% reduction,
semantic-search p95 and multi-GB savings are not demonstrated by this evidence.
Discovery timing does not measure model inference, backups, apply throughput or
semantic search. Integration fixtures validate actual merges and embedding
regeneration with deterministic inference and vector persistence mocks.

## Terminal table rendering

Victor's shared Markdown subclass overrides only its own table element registry;
Rich's global renderer is unchanged. Table cells fold long text, preserving inline
formatting and links. Panes narrower than ten columns per table field use stacked
header/value fields. `WireTimeline` removes RichLog's default 78-column minimum,
which otherwise overrides the available pane width. No fixed 90-column cap,
source-text rewriting or global Rich monkey patch is required.
RichLog retains previously rendered lines on resize; the pane-width fix applies
when new content is rendered.

Tests exercise 16, 32, 60, 90 and 140-column rendering, escaped pipes, code fences,
inline code and links, plus a mounted 32-column Textual timeline. Existing custom
tool tables remain owned by their presenters; replacing them throughout the UI
has no demonstrated benefit for this Markdown truncation defect.

## Initial local validation checkpoint

Initial implementation branch: `feature/codegraph-deduplication`, based on fetched
`origin/develop` commit `095e981bf6c0ca7eef9a73d82e4822efa4b6d1a4`. At that checkpoint work was retained
in `.worktrees/codegraph-deduplication`, before commit and publication. See the
[historical publication record](../development/completed-work.md#source-similarity-checkpoint-history)
for the rebased candidate and independent review; the
[execution ledger](graph-similarity-parity-and-roadmap.md#active-goal-execution-ledger)
owns current status.

The initial focused integration/CLI/table/timeline suite passes 35 tests and exceeds
95% coverage for the deduplication module (95.39%). Black, Ruff and MyPy on touched
production files pass. Full collection passes with 37,411 tests. A broader
affected-suite run reached 752 passing tests and 21 skips with no failures before
being interrupted after 389.79 seconds; this is a partial result, not a full
suite pass. Deterministic model and vector-provider mocks test both success and
failed persistence while using the real SQLite store and embedding pipeline.

The additional architecture/contracts, CLI, embedding and SQLite regression set
passes 159 tests with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Its first run
exposed a shutdown test that counted every identically named worker in the
process. That test now checks its own executor's worker and deliberately keeps
another owner's worker alive to catch the old global assertion. The strengthened
concurrency module passes all 30 tests. Production storage code is unchanged.
An unmodified baseline storage/concurrency run also passes its 50 tests in
isolation; the initially observed failure depended on the combined run's state.

To repeat the focused gate from the worktree with the development environment
activated (the codegraph path selects this checkout over an older editable install):

```bash
PYTHONPATH="$PWD:$PWD/victor-codegraph" python -m pytest -o addopts='' \
  tests/integration/codegraph/test_deduplication.py \
  tests/unit/ui/commands/test_graph_dedup_cli.py \
  tests/unit/ui/rendering/test_markdown_tables.py \
  tests/unit/ui/tui/test_wire_timeline.py \
  --cov=victor.core.graph_rag.dedup --cov-fail-under=90 -q
```

The subsequent source-similarity audit completed the wider affected run and
records its results, environment-specific Mermaid retry and current full
collection in [the combined validation evidence](code-similarity-audit-2026-11-01.md#initial-validation-and-continuation).
This initial increment subsequently merged in [#1657](https://github.com/anvai-labs/victor/pull/1657)
after independent review and green PR CI; the execution ledger records later increments.
The integration tests are the owner for rollback,
stale plans, disabled foreign keys, unsupported dependent state/schema, cache and
FTS consistency, shared vector references, and cancellation settlement.
