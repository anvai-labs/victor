# Windows source acquisition and qualification

The [source-similarity ledger](graph-similarity-parity-and-roadmap.md) owns task
status and release evidence. This document describes the Windows capability and
its qualification boundary; a local macOS/Linux pass does not qualify Windows.

## Source handles and namespace ownership

`source_reader.py` dispatches to typed Python/stdlib `ctypes` bindings in
`windows_source.py`. There is no compiled extension or alternate native toolchain.
The adapter opens an absolute local fixed-drive root, checks NTFS, and pins each
ancestor with read sharing but without write/delete sharing. Reparse attributes
and non-disk handles are rejected. An aliased drive root, UNC/network namespace,
non-NTFS filesystem, or unavailable primitive fails with an explicit `OSError`.

Each source path is a sequence of validated single names. `NtOpenFile` receives
the already-owned parent as `OBJECT_ATTRIBUTES.RootDirectory`, opens the reparse
point itself, and requires the expected file/directory type. Every tag is
rejected, including junctions and cloud placeholders. Alternate streams,
absolute paths, dot segments, device aliases, backslash/normalization aliases,
trailing dots/spaces and invalid characters are rejected before opening a leaf.

Only regular disk files are read. Source size is checked before acquisition;
reads retain the existing byte-budget-plus-one overflow detection. Returned
`os.stat_result` values include volume/file identity, size and nanosecond times.
The integer root handle carries its context-owned ancestor handles so inventory
never reconstructs ancestry by reopening unowned paths. All handles close on
normal completion or exceptions. Existing source hashing, coverage accounting
and async settled-worker cancellation remain in their original owners.

An open writer or a concurrent rename can prevent acquisition: that is an
explicit unreadable/incomplete result, not a reason to retry through a pathname.
Handles do not grant additional privileges and the implementation does not
attempt to defend against an administrator changing kernel/volume state.

## Inventory and Git

`windows_inventory.py` enumerates `FILE_FULL_DIR_INFO` batches using the owned
handle. Entry count, directory count, depth, UTF-8 path bytes and elapsed time are
bounded. Directory reparses fail inventory; selected leaf reparses fail source
acquisition and remain visible to coverage accounting. Skip-directory policy
matches the existing filesystem inventory.

Ordinary `.git` directories are supported. The root's pinned namespace ancestors
are inspected for the nearest Git metadata. That directory is also opened
without following reparse points and held without write/delete sharing. A cwd
pathname is obtained from the owned handle only while every namespace ancestor
remains pinned. Git receives explicit `--git-dir` and `--work-tree` arguments;
`GIT_COMMON_DIR` binds the same owned metadata and prevents a transient
`commondir` redirect. Inherited `GIT_*` selectors are stripped. Filesystem monitor
callbacks, optional locks and the untracked cache are disabled. Parent-context
checks surround inventory, and failed Git does not become filesystem coverage.

Linked-worktree `.git` files and existing `commondir` selectors fail explicitly.
They require selector identity/content ownership before support can expand. Git
is required for an identified repository. A non-Git directory uses handle
inventory regardless of Git availability. Callback-free Git and all other tools
share the Windows process owner below.

## External tool ownership

`windows_process.py` creates an unnamed Job Object with no breakaway permission
and `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. An isolated Python leader blocks on a
stdin `GO` handshake. The parent assigns the retained `Popen` process handle to
the job before sending GO, so tool descendants cannot start outside ownership.
Assignment failures terminate the waiting leader and never start the tool.

Normal descendants inherit the job. Cleanup terminates/closes the job handle;
it never kills a process found solely by PID. RSS snapshots open each reported
job PID and verify membership through its retained process handle before
measuring it. PID reuse therefore cannot redirect measurement or cleanup.

The scope string is `owned-windows-job-polled-rss-and-hard-commit`. The byte budget
has two distinct meanings: summed working-set/RSS is polled at the existing
20 ms cadence; aggregate committed memory also has a kernel-enforced job cap.
The hard commit cap can make allocation fail before RSS polling observes a
violation; the tool's resulting exit status remains a failed invocation when
nonzero. Artifact size and elapsed time retain their existing polling bounds.
These are resource and lifetime controls, not a sandbox for arbitrary hostile
code (for example, brokered process creation is outside ordinary job inheritance).

`OwnedProcess` rejects POSIX descriptor arguments on Windows. Baseline publication
still fails explicitly with `ValueError` because its rename ownership is POSIX
only; there is no unsafe Windows publication fallback.

## Qualification gates and measurements

`.github/workflows/ci-graph-windows.yml` runs actual Windows Server 2025 x64/NTFS
with CPython 3.12, 3.13 and 3.14. It exercises real handle reads, reparse rejection,
namespace and identity protection, ordinary Git ignore/subdirectory semantics,
selector rejection, job assignment failure, descendant cancellation, timeout and
artifact bounds. Portable validators and benchmark-report correctness also run
locally. Mocked APIs and native-wheel Windows checks cannot substitute for this
workflow. The focused Windows coverage configuration measures the otherwise
Ubuntu-omitted platform modules and requires 70% branch-aware coverage.
The five small Windows dispatch suites in the POSIX owners are also explicitly
excluded from Ubuntu line coverage; the Windows tests exercise those public
entry points with actual handles rather than making mocked dispatch a substitute
for OS qualification. No POSIX production suites are excluded.

The workflow preserves JUnit, coverage and acquisition benchmark artifacts. Its
benchmark acquires/hashes 4,096 files/32 MiB through both filesystem and Git
inventory, compares complete source digests and records wall time, process CPU,
throughput, parent-process CPU and sampled parent RSS. Git-child CPU is excluded;
parent RSS is not whole-job peak memory. It excludes parser/model inference. The baseline platform path was
explicitly unsupported, so this is capability qualification, not a measured
performance-refactor speedup claim. Performance ratcheting requires repeated
representative runs and a selected user-level latency target.

`ci-fast.yml` selects this reusable workflow from a complete known source diff
and joins its result and selector into `CI Success`. A required matrix must pass;
skips are accepted only after an explicit successful `required=false` result.
Missing/unknown flags, failed selection and absent/cancelled/failed matrix results
fail closed. Manual runs always require qualification. Automatic standalone
triggers are omitted to avoid duplicate Windows runs.

Before merging this capability, require all matrix jobs for the exact reviewed
candidate and the usual `CI Success` checks. Persist run URLs,
platforms, exact commit and benchmark artifact evidence in the canonical ledger.
An absent/skipped required job is not qualification.
The tested server/x64 matrix is not evidence for Windows ARM64, network volumes,
ReFS, linked worktrees, Windows baseline writes or every desktop Windows build.

The [retained platform observations](evidence/windows-source-qualification-2026-11-02.json)
bind the reviewed source head, merged PR, actual run/artifact identities and raw
file hashes. Each matrix version passes 49 tests with six POSIX-only skips.
Combined statement-plus-branch coverage is 84.25%; branch-only coverage is 71.43%.
All six synthetic 4,096-file / 32 MiB acquisition results have the same digest.
Raw reports are preserved locally outside temporary directories. Current task
status belongs to the [canonical similarity ledger](graph-similarity-parity-and-roadmap.md#active-goal-execution-ledger).

The [bounded SCIP snapshot tests](evidence/scip-windows-snapshots-2026-11-02.json)
separately exercise occurrence projection, immutable captured bytes and oversized
input rejection on actual Windows Python 3.12–3.14 with protobuf 7.36.2. Their
three tests pass per version. This qualifies that route; the scoped Rust latency
and RSS measurements remain POSIX observations.

## Primary references

- [NtOpenFile](https://learn.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntopenfile)
  and [NtCreateFile options/RootDirectory](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntcreatefile).
- [GetFileInformationByHandleEx](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getfileinformationbyhandleex)
  and [CreateFile sharing](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew).
- [Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects),
  [nested jobs](https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs)
  and [job memory limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information).
- [Git environment](https://git-scm.com/docs/git) and
  [repository selectors](https://git-scm.com/docs/gitrepository-layout).
