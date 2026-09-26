---
fep: "0036"
title: "Independent review panel with commit-bound advisory verdicts"
type: Standards Track
status: Draft
created: 2026-09-25
modified: 2026-09-25
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
---

# FEP-0036: Independent review panel with commit-bound advisory verdicts

## Summary

Add the opt-in `victor.framework.review` surface and `victor review` CLI, using
existing `AgentTeam` PARALLEL execution and reviewer roles. Provider/model choices
are explicit per reviewer; no second orchestration registry or dispatch authority
is introduced. This proposal remains Draft pending review. The implementation is
under test; no live panel acceptance or model accuracy is claimed.

Every reviewer is required: a failed or malformed response cannot be replaced by
another member's approval. Responses follow a strict JSON contract, and reviewer
identity comes from the configured team member rather than result ordering or
model-authored metadata. Reviewers receive no tools, keeping an untrusted patch
from granting command execution or workspace writes through this feature.

For pull requests, the result records the repository and immutable base/head
commit pair. Bounded GitHub reads, patch-stat checks and a final identity check
reject known incomplete or moved inputs. These safeguards constrain what the
panel reviewed; they do not establish model accuracy, satisfy human review
requirements, or authorize merging. Existing chat and team defaults remain
unchanged. Operators explicitly invoke the feature and select its reviewers.

## Motivation

The original review-panel implementation could approve a change when only one
member approved and another required member failed. It also accepted JSON
surrounded by prose, coerced malformed fields, and assigned otherwise unmatched
results by completion order. Those behaviors make an apparently successful review
ambiguous: the caller cannot tell whether every requested reviewer actually
completed the required contract for the intended patch.

Pull requests add another source of ambiguity because their heads can move while
review runs. Unbounded subprocess output and file reads can also consume resources
before the panel starts. This proposal addresses those concrete boundaries while
reusing the existing team executor and test owners. It does not substitute model
agreement for deterministic validation or a maintainer's merge decision.

## Proposed Change

`ReviewerSpec(provider, model, name="")` supplies a required panel member. Provider
and model must be nonempty trimmed strings and each resolved display identity must
be unique. The framework snapshots each configured team member ID before execution
and resolves only that ID and the matching result member ID. Display-name metadata,
completion order, and mutable `_claimed` markers cannot identify a reviewer.

`review_diff(diff, reviewers, orchestrator=...)` accepts a complete textual diff up
to 24,000 characters. Intent is bounded to 2,000 characters, leaving room within the current 40,000-character reviewer context configuration. This does not guarantee a provider never compacts its context. Empty or oversized input returns `review_incomplete` before
model dispatch. All required reviewers must succeed and return `approve`; a
successful reviewer returning `request_changes` yields that outcome even when
others abstain or fail. Failed members' output is not authoritative. A team-level
failure also prevents approval. Cancellation propagates; the configured timeout
bounds the awaited team run subject to the underlying runner's cancellation
semantics, not a guarantee that remote computation has stopped.

Each response must be exactly one JSON object containing `verdict`, `summary`,
`findings`, and `confidence`. No prose extraction, coercion, duplicate keys,
unknown fields, nonfinite numbers, booleans as numbers, or invalid finding fields
are accepted. Finding lines are positive integers or null. Confidence is numeric
in [0,1]. An approval containing major/critical findings is contradictory and
invalid. Responses over 64,000 characters are invalid. Invalid output abstains;
confidence never overrides the required-reviewer rule.

`review_pull_request` uses authenticated `gh` reads, pins repository/base/head from
GitHub PR metadata, then fetches the immutable base...head comparison. Before model
dispatch it checks file/addition/deletion counts against that metadata, refuses
binary content and PRs at GitHub's 300-file diff limit, and bounds response size.
After the panel it rechecks the metadata; changed identity invalidates the review.
The returned `ReviewVerdict` and CLI output include repository, base SHA and head
SHA. GitHub.com is supported initially; enterprise hosts fail explicitly.

This identity describes the reviewed snapshot. Callers must recheck the exact head
before using a verdict later; this API does not approve or merge GitHub PRs. A diff
file has no inferred PR identity. Neither model agreement nor JSON validity proves
correctness, security, test coverage, authorization, or completion of human review.

GitHub subprocesses have bounded reads and deadlines. On timeout/cancellation the
owned process is killed if still running, its transport closed and reap awaited
with a separate bound; stderr is discarded instead of exposing credential-helper
output. This is process cleanup, not isolation from arbitrary same-user children.

## Compatibility

```
victor review pr 1190 --repo anvai-labs/victor --reviewers zai:glm-5.3,inferflux:qwen3-coder-30b
victor review diff change.diff --reviewers zai:glm-5.3 --intent "Explain the change"
```

Exit codes are 0 approved, 1 changes requested, 2 incomplete/error. Defaults use
`Settings.provider.default_provider/default_model`; a partial explicit
`--provider`/`--model` pair is rejected. Diff files are read with a bounded byte allocation and strict UTF-8 decoding; malformed or oversized evidence exits 2 before Agent creation. Numeric execution options reject booleans, nonfinite/out-of-range temperatures and nonpositive/noninteger budgets or timeouts. The framework `Agent` owns its orchestrator
and is closed on success/failure. Existing chat, approval, tool and team defaults
remain unchanged. Reviewers explicitly receive an empty tool allowlist, excluding
role defaults such as shell, git and test. They have private team
member conversations; this is not an OS sandbox or proof of statistical independence.

## Benefits

A caller can distinguish an approved panel from an incomplete one without parsing
reviewer prose. Each recorded reviewer outcome has a configured identity, and a
failed member cannot supply an authoritative approval. The commit pair in a PR
result gives later automation a concrete value to compare before acting, while
strict input limits prevent this feature from silently reviewing a truncated
local file. Empty tool grants keep the review invocation from inheriting the
reviewer role's shell and Git capabilities. Existing team coordination remains
the execution authority, so tests can focus on the new contract boundaries
instead of maintaining a second scheduler. These benefits concern traceability
and predictable failure behavior, not proof that the review found every defect.

## Drawbacks and Alternatives

Accepting one approval among failures, extracting JSON from prose, trusting result
order, or truncating a diff and trusting the model to abstain can all produce false
approval. Those alternatives are rejected. A deterministic parser is a contract
check, not an LLM judge. Binary review, repository checkout/materialization, a live
multi-provider evaluation, durable reviewer recovery and automated merge authority
are outside this increment. API diff completeness checks establish textual scope,
not that reviewers obtained every dependency or understood all affected code.
Requiring all reviewers to approve also makes provider outages visible as incomplete
reviews and may reduce availability. Operators can choose a smaller panel, but the
implementation must not silently drop a required member to obtain approval.

## Unresolved Questions

Live multi-provider accuracy and cost remain unmeasured. Supporting enterprise
GitHub hosts, binary changes, larger patches, or repository context requires
separate contracts and validation. Durable recovery of an interrupted panel and
safe application of an old verdict to a later action are also outside this
increment; no automatic merge or authorization path is proposed here.

## Migration Path

This is an opt-in public surface with no migration for existing chat or team
callers. The unreleased draft panel's permissive output handling is intentionally
replaced: callers must supply complete bounded inputs and treat incomplete results
as unsuccessful review. Existing published releases are not changed. Users may
continue their current review process without invoking the new CLI or API.

## Implementation Plan

Extend `tests/unit/framework/test_review.py` for strict response negatives, required
reviewer aggregation, exact identity, truncation refusal, immutable PR snapshots,
server truncation, cancellation and subprocess cleanup. Extend the existing CLI
owner for actual Settings shape, malformed reviewer lists, exit codes and Agent
cleanup. Use existing team tests for coordination contracts. No new mirror suite,
model calls, hosted review posting, or service mutation is needed for these tests.
Run both workflow FEP-validation paths, the numbering check, changed-file test
selection, full collection, formatting, typing and documentation drift checks
before publication. Independent review and the repository's required CI gates
remain prerequisites; passing these checks does not establish live panel acceptance.

## References

- [FEP process](../docs/FEP_PROCESS.md)
- [GitHub repository and diff limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)
- [GitHub compare commits API](https://docs.github.com/en/rest/commits/commits#compare-two-commits)
