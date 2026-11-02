# CodeGraph deduplication and similarity: independent review evidence

Review base: `b3f09333aef5524a38a55fd9aee6f33973b82fa8` (`origin/develop`).
Initial candidate: `23f7c3a0b7b16e5f7ebf761134c891eca1dda135`.
The [parity assessment](graph-similarity-parity-and-roadmap.md) owns the capability
assessment; exact final-commit review attestations live in Git metadata and PR
validation evidence, not a self-referential source commit hash.

## Initial independent sessions

Three read-only sessions reviewed storage/repair, evidence/cache/process handling,
and AST/portable/JNI/TUI behavior. Storage review was clean: 46 tests and separate
malformed-edge rollback, fabricated-plan rejection, repeated-cancellation and
persisted-vector/pending-marker retry experiments passed. No reviewer edited
source or published an attestation for an unreviewed revision.

The other sessions reproduced eight P2 findings despite passing existing tests.
Each was corrected before the first CI-triggering push:

| Finding | Correction | Regression evidence |
| --- | --- | --- |
| Corrupt body databases escape hybrid handling | Catch SQLite errors in the body channel; close failed initialization connections | `test_corrupt_body_database_is_local_to_body_channel`, `test_locked_body_database_is_local_to_body_channel` |
| Body function budgets suppress all hybrid channels | Separate body eligibility limits from hybrid collection; preserve tokens after collection failure | `test_body_budget_does_not_suppress_syntax_and_token_channels`, `test_shared_collection_failure_still_allows_tokens` |
| Children survive successful direct-parent exit | Live isolated session leader, separate atomic status, cleanup before reaping | `test_normal_parent_exit_cleans_up_its_child`, `test_cancellation_after_spawn_reaps_owned_group`, `test_fast_output_still_obeys_artifact_limit` |
| Source links escape the project | Descriptor-relative no-follow traversal and regular-file validation; fail closed without support | `test_external_symlink_and_fifo_cannot_become_source_units`, `test_ancestor_replacement_cannot_redirect_an_open_directory`, `test_leaf_symlink_replacement_is_rejected`, `test_token_snapshot_rejects_external_source_symlink` |
| Relative backend path fails after cwd change | Resolve executable before verification and scratch execution | `test_relative_backend_is_resolved_before_changing_cwd` |
| Compound assignment loses operator | Preserve operator and single target evaluation | `test_compound_assignment_preserves_the_operator` for C/C++/Java |
| Python subtree/exact streams flatten branch roles | Version v4 field-labelled AST containers and field layout | `test_python_branch_field_roles_are_not_erased`, `test_python_keyword_target_name_is_preserved` |
| Annotation string becomes false JNI native evidence | Require the direct modifier token | `test_annotation_string_is_not_a_native_modifier` |

Darwin rejects signaling a zombie-only group with EPERM. The live stdlib launcher
avoids treating that as successful cleanup or ignoring permission errors. It runs
fixed controller code under `-I -S`; source/configuration text is never interpreted
as Python. Descendants creating independent sessions are outside the process-group
cleanup contract. Secure source reads currently require POSIX openat primitives;
Windows needs a separately qualified handle-relative adapter.

## Local gate evidence

The repository's changed-test selector now maps the existing owners for new
source modules, including the dedup integration owner. No gate or threshold was
weakened. A CI-shaped run passed 796 tests with 86% diff coverage, and the invariant
and selector guards passed 319 tests. Global strict MyPy passed 2,209 source files;
Bandit reported zero blocking high-severity/high-confidence findings. The pinned
duplication ratchet remained at its 6,501-line baseline. Updated post-review
checks and the exact reviewed SHA are recorded in the PR and commit-bound local
attestation before publication. These results do not claim full-suite, Windows,
installed-wheel, release or live-provider acceptance.
