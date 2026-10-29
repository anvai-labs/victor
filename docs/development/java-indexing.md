# Java indexing and SCIP qualification

Victor's shared codegraph extracts Java classes, interfaces, enums, records,
annotation types/members and explicit constructors, including compact record
constructors. Implicit compiler-generated members are not fabricated as source
symbols. Constructor names normalize to `constructor`, matching the SCIP provider;
source signatures and containing types preserve overload identity.

`scip-java` remains opt-in. Its current producer uses `method().` and
`method(+N).` descriptors. The ordinal is not arity or a signature. The harvester
matches file/name/kind/scope, then uses the definition's source line to separate
multiple candidates with complete bounds. Same-line, overlapping or unbounded
candidates remain unjoined. Interface/enum/struct/constructor parents participate
in scope reconstruction; no fallback flattens ambiguous symbols.

## Toolchain

The official [scip-java v0.13.2 release](https://github.com/scip-code/scip-java/releases/tag/v0.13.2)
was current when checked on 2026-10-29. Verify its published SHA-256 before using
it. That release's executable and SCIP metadata report `0.0.0-SNAPSHOT`; identify
it by release URL and digest rather than trusting its version string.

Java 17, 21 and 25 are supported by that producer. Use a JDK, including `javac`,
not only a JRE. See the [tagged upstream guide](https://github.com/scip-code/scip-java/blob/v0.13.2/docs/getting-started.md).
Gradle and Maven are automatically configured. Modern Bazel roots with
`MODULE.bazel` or `WORKSPACE.bazel` are detected; the adapter supplies the absolute
indexer path required by its aspect. This argv path is unit-tested, not a new real
Bazel qualification. Scala/sbt, Mill and Ant-only roots skip before execution.
Bazel aspect authoring and manual configurations remain upstream responsibilities.

Compiler indexing executes repository build logic and may clean build outputs.
It needs explicit operator authorization and a disposable source copy. For an
untrusted checkout, use an isolated process/filesystem with a cleared environment,
no ambient credentials or user configuration, dedicated caches, a fixed reviewed
build invocation and a timeout. Syntax-only indexing does not compile the project.
Never turn model-selected commands into an index recipe.

## Measured Iceberg experiment

Pinned Apache Iceberg commit: `2ebe838e314e5f1ad8c5fb16662e7434fc11de38`.
This is a substantial multi-module corpus: 6,621 tracked files and 5,999 Java
files (53,291,142 Java source bytes). The comparison uses identical source bytes
and the same retained compiler index; no generated source becomes pinned evidence.

| Measurement | Before | Improved |
| --- | ---: | ---: |
| Syntax files processed | 6,215 | 6,215 |
| Tracked Java files represented | 5,999 | 5,999 |
| Partial syntax files, all languages | 26 | 24 |
| Persisted nodes | 82,927 | 88,306 |
| Persisted syntax edges | 135,533 | 142,248 |
| Joined compiler REFERENCES edges | 30,591 | 39,635 |
| Unjoined definitions | 2,309 | 354 |
| References to unjoined definitions | 4,801 | 1,679 |
| Ownerless references | 3,149 | 3,158 |

The scoped compiler build uses verified scip-java v0.13.2 and Temurin JDK
21.0.13+9, running Gradle 8.14.6 in a disposable isolated copy with dedicated
caches and no user credentials. API/common/core compilation also builds
bundled-guava. It succeeds in 79.4 seconds after correcting the initial isolated
copy's missing Git version metadata and replacing the system's JRE-only Java.
Build timing includes warm dependency caches from unsuccessful setup attempts;
it is not a cold-install or performance claim.

The retained compiler artifact has 1,013 documents and 275,559 occurrences:
678 core, 330 API, four common and one bundled-guava document. Forty-nine are
generated sources outside the tracked syntax corpus and remain explicitly
out of corpus. This is **scoped compiler coverage**, not a whole-Iceberg SCIP
qualification. The improved whole-source syntax index still has 24 partial
files (22 Java). More joined edges do not establish source correctness,
complete call-graph coverage, tutorial quality or a speedup.

Validation retains positive parser/constructor/overload tests and negative tests
for unsupported build roots, malformed ordinals, unknown/overlapping/outside
bounds and absent SCIP ranges. Real Bazel, all-module compilation and generated
source admission are not qualified here. Ignored local artifacts and complete
commands live in the RepoReel RR-051 experiment; no source clones, compiler dumps
or dependency caches belong in Git.
