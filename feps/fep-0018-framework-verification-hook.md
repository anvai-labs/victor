---
fep: "0018"
title: "Framework verification hook — verify the agent's work after COMPLETE + retry on failure"
type: Standards Track
status: Draft
created: 2026-07-15
modified: 2026-09-20
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/0018
---

# FEP-0018: Framework verification hook

## Summary

Add a **framework-level verification hook** to the agentic loop: after the agent claims COMPLETE, run a pluggable `Verifier` that checks the agent's work against real-world feedback (tests, lint, dry-run). If verification fails, inject the result as feedback and re-enter the loop (bounded retries). Today this capability is **eval-only** (the benchmark adapter's verify-and-retry gate). FEP-0018 promotes it to the loop itself so ANY agent session benefits.

## Motivation

The verify-and-retry capability proved its value in SWE-bench eval: it achieved **97/97 VERIFIED** on one astropy task — the agent saw "9/10 tests passed, test_foo failed", fixed it on retry, and achieved a full pass. But:
- It lives in `agent_adapter.py` (eval-only) — interactive chat, workflows, CLI get no verification.
- It's wired by `benchmark.py`'s verify_fn closure — SWE-bench-specific.
- `StreamingChatExecutor` has no verify gate at all.

FEP-0018 promotes the capability to the agentic loop's DECIDE phase, making it available to every agent session.

## Proposed Change

### `Verifier` protocol (`victor/framework/verification.py`)
A `@runtime_checkable` Protocol with `async def verify(*, workspace, state) -> VerificationResult`. `VerificationResult` carries `passed/total/raw_output/feedback` + an `is_verified` property. Zero victor deps (stdlib only).

### Verify gate in the DECIDE phase (`victor/framework/agentic_loop.py`)
After DECIDE=COMPLETE + backslide guard, the shared `_maybe_verify_and_retry`
gate verifies every completion claim when a verifier is supplied. This includes
the final retry; `max_verify_retries=0` means one check without a repair retry.
A failed or empty verification, or verifier exception, sets the evaluation to
RETRY while retries remain and FAIL when exhausted. Failure also sets the reward
score to zero. Hitting the outer iteration limit while verification is pending
cannot retain a COMPLETE evaluation.

Both `run()` and `run_streaming()` use this gate. Structured counts and verification
status are retained in `state["verification"]` and evaluation metadata. Retry budgets
reset on each run. When a verifier is supplied, buffered execution bypasses semantic
response caching because cached prose is not evidence about current artifacts.
StateGraph execution and the older `stream()` iteration API reject a configured
verifier explicitly until those paths implement the verification contract.

### Loop constructor: `verifier` + `max_verify_retries` params
```python
AgenticLoop(..., verifier: Optional[Verifier] = None, max_verify_retries: int = 0)
```
Default None/0 = no behavior change. Supplying a verifier with zero retries still
checks once and reports failed verification honestly.

### Built-in structured process acceptance (G36)

`LocalTestVerifier` uses the existing runner detector but currently accepts only
pytest commands (`python -m pytest` or a pytest executable). It appends a fresh
runner-owned JUnit destination and validates the report tree, test cases and
outcome totals. Missing, malformed, contradictory, empty and skipped-only reports
do not verify work. Other ecosystems require a custom `Verifier` until explicit
structured adapters are implemented; detection failures never substitute a runner.

A nonempty valid report contributes its executed test-case checks plus one explicit
process-exit check. For example, one passing test with exit zero is 2/2 checks;
one passing test with exit one is 1/2. Feedback retains the separate test counts
and exit status. Skips are excluded from executed-test counts; failure/error
outcomes cannot be hidden by a conflicting skip. `LintVerifier` contributes one
process-status check and treats diagnostic text only as diagnostics.

Buffered verifier processes retain bounded diagnostic tails, use file output to
avoid inherited-pipe hangs, and have a separate one-second cleanup deadline.
Timeout and incomplete cleanup return failure statuses; cancellation remains an
exception with cleanup notes. Observed exits are never signalled. This is bounded
verification, not detached-child containment or proof that member-authored tests
cover the task. The `Verifier`/`VerificationResult` public signatures are unchanged;
normal runs without a configured verifier keep their behavior.

### Helpers: `_run_verification` + `_inject_verify_feedback`
- `_run_verification(state)`: calls `self._verifier.verify(workspace=..., state=state)`, returns `VerificationResult`.
- `_inject_verify_feedback(result)`: injects the result as a user message via `turn_executor._chat_context.add_message()` (mirrors `_inject_decide_nudges`).

## Benefits

- **Any agent session** can now verify + retry, not just benchmark eval.
- **Streaming parity**: `run_streaming()` gets the verify gate for free (it's in the loop, not the adapter).
- **Pluggable**: any `Verifier` implementation (container test, local test, lint, dry-run, citation-check) can be wired without loop changes.
- **Backward-compatible**: `verifier=None` (default) = zero behavior change.

## Drawbacks and Alternatives

- **Wiring complexity**: the verifier must be threaded through the orchestrator → turn_executor → AgenticLoop injection path. Phase 1 lands the protocol + loop gate (backward-compatible); Phase 2 wires the SWE-bench verifier + removes the adapter's duplicate gate.
- **Alternative considered**: keep verification in the adapter (status quo). Rejected: the adapter is eval-specific; the capability is generic.
- **Alternative considered**: a separate "verification middleware." Rejected: the DECIDE gate is the natural place (it's where the loop decides to exit); a middleware adds indirection.

## Unresolved Questions

- **Verifier injection path**: how to thread the verifier from SessionConfig/profile → AgenticLoop for non-eval sessions (interactive chat, workflows). Phase 2.
- **Adaptive retry budget**: should `max_verify_retries` adapt based on progress (more retries for near-misses like 23/24)? Phase 3.
- **Multi-verifier composition**: should multiple verifiers run in sequence (test + lint + type-check)? Phase 3.

## Implementation Plan

- **Phase 1 (this PR)**: Protocol + loop gate (run + run_streaming) + helpers. Backward-compatible (verifier=None default). The adapter's existing gate stays until Phase 2.
- **Phase 2 (follow-up)**: Migrate the SWE-bench verifier as `ContainerTestVerifier(Verifier)`. Wire the benchmark to pass it to the loop. Remove the adapter's duplicate gate (one gate, not two).
- **Phase 3 (follow-up)**: Generic verifiers (LocalTestVerifier, LintVerifier, DryRunVerifier) + SessionConfig/CLI surface (`--verify` flag) + adaptive retry budget.

## Migration Path

Additive and non-breaking. The protocol + loop gate are dormant by default (verifier=None). The adapter's existing gate continues to work. Phase 2 migrates the SWE-bench path to the loop gate.

## Compatibility

- No public API change (new optional params, new module).
- All existing tests pass (91 unit + 28 streaming parity batteries verified).
- The adapter's verify gate (agent_adapter.py:1309-1338) is unchanged in Phase 1.

## References

- FEP-0017 (prompt-optimization reward loop) — the verify gate runs before the reward emission.
- `victor/framework/workspace.py` — `workspace_git_diff` for the ground-truth patch.
- The SWE-bench eval proving the concept: `Turn 2 verify: 97/97 (VERIFIED)`.
