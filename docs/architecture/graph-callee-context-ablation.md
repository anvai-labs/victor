# Callee-context ablation

The [canonical similarity ledger](graph-similarity-parity-and-roadmap.md) owns
task and acceptance status. This experiment measures whether resolved call-target
identities distinguish controlled counterexamples that normalized syntax merges,
and reconnect differently spelled import aliases. It precedes graph-feature
ranking integration. Weighted fusion remains disabled.

## Inputs and intended observation

[Frozen authored controls](https://github.com/anvai-labs/victor/blob/develop/tests/fixtures/graph_context/manifest.json) pin
five Python files, 1,876 source bytes, 12 compared functions and eight pairs.
The oracle has three positives, four negatives and one unknown. Labels were
frozen before scoring, and independent source review accepted them unchanged.
The domain requires the exact frozen classes/functions, built-in values and no
subclass overrides or monkeypatching. For example, `Registry.get` returns string
length while `Cache.get` negates it; their otherwise identical callers differ
for nonempty strings. The source formulas supply the restricted oracle, rather
than method names or a detector score.

The old audit's `scip_different_targets` probe is retained separately. Its injected
Registry/Cache identities change one exact syntax family into template/near
evidence. That is a hand-authored identity control, not compiler-produced output.

The new fixture also contains actual [scip-python](https://github.com/sourcegraph/scip-python)
0.6.8 output from its Pyright-based static indexer. The 20,787-byte raw SCIP hash is
`397125671f0645bfa656202d9a2b306ff257aa5362d1a62b5d431d37589904e3`.
It is distinct from the curator-controlled leg. Static indexing does not compile
or execute these functions, verify runtime dispatch, establish effects/lifetimes,
or prove behavioral equivalence.

## Representation and missing evidence

`scripts/benchmark_callee_context.py` captures only explicitly listed source bytes,
checks every source hash and exact function span, then uses the production Python
normalizer and subtree fingerprints. Three legs retain the same pair universe:
syntax, syntax with curator-declared call-target leaves, and syntax with actual
SCIP call-target leaves. Non-callee type/data identities are excluded from the
ablation. Fully qualified Python function/method targets qualify this narrow
projection; local placeholders, callable values, constructor guesses, missing or
ambiguous targets remain unresolved. A caller with an unresolved call abstains
from the context leg. Functions with no calls retain unchanged syntax evidence
and an explicit `no_call_sites` status.

The exact-match diagnostic uses the pre-existing normalized-token identity rule.
Subtree set-Jaccard scores and the entire labelled precision-at-k curve are also
retained. Cutoff ties use expected uniform ordering; no chosen review budget,
threshold tuning or score fusion optimizes these controls. Unknown labels are
excluded from accuracy counts, and absent evidence is never a true negative.
Scores apply only to these requested pairs, rather than repository retrieval.

The [retained observations](evidence/callee-context-ablation-2026-11-02.json)
bind the current consumer to `ca797a4e8`, the observer's exact source hash and all
frozen inputs. Controlled identities and actual index output agree on each pair's
score/decision while retaining distinct target namespaces and provenance:

| Representation | Labelled coverage | TP / FP / FN / TN | Exact-rule precision | Recall |
| --- | --- | --- | --- | --- |
| Syntax | 7/7 | 2 / 3 / 1 / 1 | 0.400 | 0.667 |
| Curator call-target control | 7/7 | 3 / 1 / 0 / 3 | 0.750 | 1.000 |
| Actual SCIP call targets | 7/7 | 3 / 1 / 0 / 3 | 0.750 | 1.000 |

The eighth pair has an unknown oracle and unresolved `Any` targets, so it does
not become a labelled negative. The complete rank curves preserve every budget
and cutoff ties. These numbers describe this diagnostic universe; they are not
calibrated probabilities or evidence of general semantic-deduplication quality.

The recorded compiler documents omit text. Their binding checks raw index SHA-256
and all captured source hashes against the generation sidecar; a source mismatch
abstains independently of timestamps. The sidecar remains curator provenance,
not cryptographic authentication of an indexer's source read. The fixture's ASCII
identifiers avoid ambiguity from its unspecified per-document position encoding.
Duplicate documents, producer conflicts, byte/occurrence-budget violations and
invalid manifests reject rather than selecting a convenient observation.

## Bounds and replay

Budgets are 64 KiB per source/JSON file, 128 KiB total source, 32 files/documents,
64 compared units, 256 requested pairs, 1,024 call sites per compared unit,
20,000 indexed occurrences and 2 MiB each for index and output. Source capture
uses the existing owned reader. Implementation and manifest changes during a
measurement invalidate it. Reports contain identities, hashes and diagnostics.

Report publication owns its parent before measurement and uses descriptor-relative
exclusive creation. A swapped parent, protected input, existing destination or
unsupported publication primitive fails closed. Temporary descriptors close on
exceptions/cancellation; cleanup removes entries only while their inode identity
is still owned. Publication is currently POSIX-only and introduces no Windows
write fallback.

The [fixture instructions](https://github.com/anvai-labs/victor/blob/develop/tests/fixtures/graph_context/README.md) replay the
frozen index without an indexer or model download. The CI test selector maps both
the script and every fixture change to its existing ablation owner and the shared
`test_scip_documents.py` owner. The latter selects the graph extra's declared
protobuf runtime and mandatory-reader flag. The actual-fixture tests fail under
that flag if the runtime is missing, rather than passing through optional skips.

Actual generation used the already installed `scip-python 0.6.8`, an explicit
project name/version and an isolated copy containing only the five listed sources
and recorded `pyrightconfig.json`. The existing Python 3.12.6 environment was
selected for the child process; user-site packages and inherited `PYTHONPATH`
were disabled. `OwnedProcess` bounded generation to 45 seconds, 512 MiB polled
process-group RSS, 128 KiB stdout/stderr and 2 MiB output. The successful run took
29.799 seconds wall and 8.436 seconds parent CPU. That CPU figure excludes the
indexer/children. This is one local generation observation, not a scalability or
latency target. Regeneration can change metadata paths and the raw index hash;
capture a new binding instead of relabelling the frozen producer record.

Five fresh-process replay observations retain identical sources, targets and pair
geometry on one ARM64 macOS host with Python 3.12.6 / protobuf 6.33.6. Median full
CLI wall time is 430.150 ms, including startup/imports and exclusive publication.
The observer's capture/extraction/scoring scope is 72.093 ms wall / 39.922 ms CPU;
index capture/projection is 18.229 ms. Median syntax/controlled/indexed extraction
legs are 6.051 / 3.444 / 3.945 ms. Their fixed order and warming mean these are
cost observations, not a claim that context accelerates extraction. Lifetime
process peak RSS has median 56,360,960 bytes and includes imports and every leg;
it does not isolate incremental feature memory. Generation and replay are
separate scopes, and no full-project latency target is qualified.
The shared host's scheduling and cold-cache state are uncontrolled; all five raw
observations are retained without selecting a favorable run.

## Limits

The controlled and actual-index legs preserve a literal-change false positive:
the production normalizer erases literal values, so a matching callee does not
recover that lost distinction. The `Any` receiver remains unknown. Syntax/context
precision on seven correlated labelled controls cannot qualify production quality.
Def/use, effects, compiler CFG/DDG, cross-language type/domain aliases, model
calibration and independent-project retrieval remain separate evidence needs.
No score certifies semantic equivalence, extraction safety or ABI compatibility.
