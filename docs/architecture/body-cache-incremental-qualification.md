# Body-cache incremental and exact-retrieval qualification

This protocol qualifies cache reuse and exact threshold retrieval on one bounded,
immutable actual-source workload. Current task status and merge evidence belong to
the [similarity ledger](graph-similarity-parity-and-roadmap.md#active-goal-execution-ledger).
It does not measure detector accuracy or claim incremental syntax collection.

`scripts/qualify_body_incremental.py` exports the 40-file source manifest from
Git revision `7b4942c910ced3d9ce8f6daaac65aa0cee64285b`. The original manifest is
[body-cache provenance evidence](evidence/body-cache-provenance-2026-11-02.json).
The workload contains 234 functions and 233 distinct model inputs; repeated
bodies remain separate functions when validating all 27,261 requested pairs.

## Acceptance and measurement scope

Each observation runs in a fresh bounded process. Source collection, requested
pair scoring and global retrieval have separate wall and CPU timers. Global
retrieval follows requested scoring, so its vector cache is warm even when the
observation's requested stage starts cold. A fresh global encoder must load no
model, infer no windows and hit every distinct active input. Process peak RSS
covers the lifetime of the worker, rather than the global stage alone.

The driver performs a warm repeat, a comment edit inside one leaf function, an
unstaged deletion, a staged deletion, restoration, rename and rename restoration.
Each stage checks its independently expected selected-file hashes, function
count and source-coverage status. The unstaged deletion must report incomplete
coverage; it is retained as a negative and receives no global certificate.
Valid stages also receive separate fresh-cache cold references.

For byte-unchanged source files, the function cohort, cached float32 vector bytes,
window spans and common requested scores must remain exact. The controlled edit
must invalidate exactly one independently captured model input. A staged delete,
restore or rename must not load a model or infer windows. Cached rows for old
inputs may remain stored, but they must not enter active output cohorts or retain
stale file locations.

`scripts/body_qualification_metrics.py` verifies complete ordered unit manifests
and the full function-index pair universe before grouping repeated input hashes.
It binds requested and global producer/input profiles and global options, checks
cache checksums, declared dimensions, spans and finite positive float32 norms,
then builds an independent float64 dense cosine matrix. The oracle validates
every requested score and independently constructs connected components for
global threshold 0.9 with same-file pairs excluded. Public family membership,
edge counts and score extrema must agree within 1e-6. The API does not expose raw
matching edge identities, so this is not an edge-list identity proof.

Historical cache rows keep their producer identity. A producer mismatch must
cause inference, even if the resulting vectors happen to match older bytes.
Fresh-cache references are new inference measurements; validation of retained
reports is reported separately and never relabeled as fresh inference.

## Ownership and bounds

The qualification driver is a POSIX-only measurement tool; production Windows
source support is qualified separately. `scripts/body_qualification_io.py`
exclusively reserves a mode-0700 output directory beneath a pinned parent and
mutates source copies through no-follow directory descriptors. Source paths cannot install Git
metadata, attributes or ignore controls. Owned Git commands strip inherited Git
selectors, isolate system/global configuration, disable callbacks and maintenance,
and select their owned work tree explicitly. Source files are never executed.

Worker cwd is inherited from the pinned private directory. Its file-backed
production cache remains a single relative SQLite leaf, including its sidecars;
replacing an ancestor pathname cannot redirect SQLite writes to the source or
seed tree. Completion requires a final exclusive `QUALIFICATION_COMPLETE` marker
and verification of every pinned ancestor and the output/source directories.
An existing destination or completion marker is never overwritten. The valid
marker contains exactly `victor-incremental-body-qualification-v2` followed by a
newline. Incomplete directories are retained without a valid marker for diagnosis.
The original source repository and cache seed are read only. Seed caches must be
closed DELETE-journal snapshots without sidecars at capture and recheck;
their bounded bytes are deserialized into memory before SQLite inspects them.

Limits are 100 source files, 4 MiB per exported file, 64 MiB total source bytes,
512 KiB manifest, 16 MiB cache input, 2,000 cache rows, a five-second SQLite query
deadline, 180 seconds and 4 GiB polled process-tree RSS per inference worker,
32 MiB worker output and 256 KiB stderr. Child groups are settled on budget
failure. These are qualification bounds, not production capacity guarantees.
The focused tests cover callback isolation, source/publication aliases, bounded
input and worker failures, malformed cohorts and cache corruption.

## Reproduction

Use the repository's offline model environment and an absent output directory
beneath an existing owned parent:

```sh
PYTHONPATH=.:victor-codegraph:victor-contracts \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/qualify_body_incremental.py \
  --repository . \
  --manifest docs/architecture/evidence/body-cache-provenance-2026-11-02.json \
  --revision 7b4942c910ced3d9ce8f6daaac65aa0cee64285b \
  --seed-cache /owned/checkpointed-historical-body.sqlite \
  --output /owned/absent-qualification-output
```

Raw reports, cache copies and source snapshots are retained outside temporary
directories. Aggregate evidence must identify observer hashes, producer identity,
source hashes, numerical tolerance and actual timing scope. Replicated-vector
capacity probes are labeled synthetic and cannot establish independent-source
accuracy, retrieval generalization or a production scaling threshold. No ANN,
native extension or production retrieval algorithm change is introduced here.
