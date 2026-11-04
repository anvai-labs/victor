# Similarity evaluation inputs

`controls.json` and its two source directories are authored protocol controls.
They deliberately include renamed arithmetic, changed operators/literals, and
Python/Rust and TypeScript/Rust numeric-domain negatives. Related templates occur
in both fixture partitions: they do not constitute held-out accuracy evidence.
Source filenames start with `control_` so pytest does not collect them as tests.
The declared `domain`, `rationale` and `origin` are part of each label's meaning.

`repository-pilot.json` records source-inspected labels and hashes at immutable
Victor and ProximaDB revisions. ProximaDB source is not vendored. Supply explicit
source roots containing the recorded bytes. The pilot has only eight pairs;
structural labels do not assert runtime, floating-point or persistence equivalence.

`independent-projects.json` adds 24 source-inspected Python pairs from independently
maintained Requests and Flask releases. Requests is the calibration project and
Flask is held out; each has six positives and six hard negatives. All labels were
frozen before scoring. Their domains describe local operation patterns and allow
the specified identifier/member/constant substitutions; they make no behavioral
equivalence or source-copy claim. Several positives are small sibling wrappers,
so the 24 pairs are correlated and do not represent 24 independent lineages.

No third-party source is vendored. The
[retained observations](../../../docs/architecture/evidence/graph-independent-projects-2026-11-02.json)
list six immutable primary-repository raw URLs, byte counts and SHA-256 hashes.
Download only those files into separate `requests` and `flask` roots, preserving
their project-relative paths, then run the command below. Do not import, install
or execute those source files. Source snapshots from the recorded run persist in
the canonical workspace's ignored
`.victor/sandbox/graph-project-calibration-20261102/{requests,flask}` roots,
independent of the implementation worktree's lifetime. The original frozen
manifest is retained alongside those roots and all copied bytes were hash-verified.

```bash
PYTHONPATH=.:victor-codegraph python scripts/evaluate_graph_similarity.py \
  --corpus tests/fixtures/graph_similarity/independent-projects.json \
  --project requests=/path/to/requests --project flask=/path/to/flask \
  --channel syntax --output /path/to/independent-syntax.json
```

Repeat with `--channel portable` to retain unsupported-source abstentions (exit 1
is expected for the recorded profile). For body measurements, set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, use `--channel body`, and supply an
exclusive `--cache /path/to/derived-body.sqlite`. Use an absent cache for a cold
run, then reuse it for the warm run. The pinned BGE profile has complete scoring
but no feasible calibration threshold at the recorded precision/support target;
`--require-decisions` therefore fails. No model download is necessary or allowed.

See [the protocol and commands](../../../docs/architecture/graph-similarity-evaluation.md).
The fixture prefix has explicit changed-test owners, including JSON and Rust/
TypeScript/Java/C++ source edits. The full portable control test is optional when
its grammar wheels are absent; Python-only command/validation tests remain active.
