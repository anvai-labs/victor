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

See [the protocol and commands](../../../docs/architecture/graph-similarity-evaluation.md).
The fixture prefix has explicit changed-test owners, including JSON and Rust/
TypeScript/Java/C++ source edits. The full portable control test is optional when
its grammar wheels are absent; Python-only command/validation tests remain active.
