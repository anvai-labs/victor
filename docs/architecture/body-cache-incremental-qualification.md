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

## November 2 actual-source observation

The [aggregate evidence](evidence/body-incremental-qualification-2026-11-02.json)
records fresh inference at local measurement commit `79bc99e5417`, based on
merged develop `ca797a4e8fb`. The ten bound observer/engine files and raw reports,
cache copies, source snapshots and historical observations are preserved in the
local `.victor/sandbox/graph-incremental-qualification-20261102` archive. Its
manifest covers 282 files and 74,174,678 bytes. The earlier observer remains
explicitly historical; these results come from the frozen, independently reviewed
observer with all reported acquisition and validation findings fixed.

The environment was Python 3.12.16 on Darwin arm64 with offline BGE small weights.
Producer `459e313a56d8` did not reuse any of the 233 historical `69592c94a263`
producer rows. The baseline loaded one model and inferred 274 windows for 233
distinct inputs. Its requested-score digest is
`a80561ed29ba2d854b3ff6927cc5f1ad14d0029479701ac84ef91208c5067200`.

| State | Functions | Cache hits | Inferred windows | Collection wall (s) | Requested wall (s) | Warm global wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline with foreign seed | 234 | 0 | 274 | 0.635 | 35.310 | 0.030 |
| Warm repeat | 234 | 233 | 0 | 0.509 | 0.066 | 0.032 |
| Comment edit | 234 | 232 | 1 | 0.782 | 7.692 | 0.033 |
| Unstaged deletion, incomplete | 228 | 227 | 0 | 0.549 | 0.062 | 0.029 |
| Staged deletion | 228 | 227 | 0 | 0.494 | 0.062 | 0.030 |
| Restore | 234 | 233 | 0 | 0.526 | 0.064 | 0.033 |
| Rename | 234 | 233 | 0 | 0.541 | 0.063 | 0.030 |
| Rename restoration | 234 | 233 | 0 | 0.535 | 0.066 | 0.033 |

Only the baseline and edit loaded a model in the incremental sequence. Every
global call used a fresh encoder and reused all active vectors with zero windows
and model loads. All six valid states passed separate fresh-cache cold references;
the unstaged deletion received no certificate. Shared vector bytes, spans and
requested scores stayed exact. The edit's maximum cold-reference score difference
was `1.788139e-7`, and its maximum vector-component difference was `8.568168e-8`;
all other cold-reference differences were zero. Restoration recovered the exact
baseline score digest, and rename output contained no stale original locations.

The full 234-function global oracle found 15 families and 20 matching edges.
Its maximum requested-score difference from independent float64 dense cosines
was `2.122872e-7`; the closest score was `0.00026345` from threshold 0.9.
This supports numerical agreement for this workload. It does not establish a
general accuracy threshold or raw edge-list identity. Process-lifetime peak RSS
was 1,064,697,856 bytes for the baseline and 90,079,232 bytes for the warm repeat.

Three sequential capacity observations per size replicated the final validated
233-by-384 vector fixture at synthetic locations. At 1,000 records, median wall
and CPU were 0.028837/0.028782 seconds with 69,992,448-byte peak RSS. At 5,000,
they were 0.300171/0.304849 seconds with 146,128,896-byte peak RSS. These are
capacity-only measurements with repeated vectors, rather than 5,000 independent
functions. They motivate no ANN or native rewrite and demonstrate no algorithmic
speedup: the production retrieval algorithm is unchanged.

After G1 merged, the [separate consumer revalidation](evidence/body-incremental-revalidation-2026-11-02.json)
checked current develop `1c6fe96e581` against the retained original reports and
cache snapshots. All ten bound observer/engine hashes were unchanged. All 14
reports passed their coverage/cohort checks, including 13 complete dense-oracle
checks and the explicit incomplete-deletion negative. This replay performed no
model inference or model-cache writes; it does not change the original run's
`ca797a4e8fb` baseline or its timings.
