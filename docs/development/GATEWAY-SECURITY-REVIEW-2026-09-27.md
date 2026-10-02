# Gateway client consolidation review — 2026-09-27

This candidate adds virtual-key or renewable OIDC gateway credentials and terminal
policy decisions without exposing upstream credentials to individual clients.
It is paired with the Sandhi identity/policy gateway candidate; it does not switch
message-hub traffic or activate a production route.

## Independent review findings and fixes

The pre-push review reproduced and closed:

- Explicit virtual-key configuration without a URL silently selected direct mode.
  Credential/scope configuration now fails closed when a gateway URL is absent.
- Outer recovery retried denials: subagents, retry helpers, empty-response and
  tool-failure completion, summary helpers, and the optional rubric judge.
  They preserve the same ProviderPolicyError and receipt, with no additional call.
- AgenticLoop and StateGraph changed denials into empty/failure results. ACT,
  node execution, graph runtime and graph adapters now propagate the terminal
  exception; real graph run/stream regressions cover the entire path.
- Gateway detection missed factory-created ManagedProvider wrappers. A shared,
  bounded, cycle-safe boundary check handles managed and resilient wrappers.

Regression tests failed before the fixes (9 initial cases, 5 runtime cases,
8 graph/wrapper cases and the optional-judge case). An independent re-review
closed the findings and separately ran 74 focused tests plus the rubric checks.
The exact final commit is bound in the local review attestation and PR body.

## Validation and scope

The consolidation pass ran 511 affected tests, then the full CI changed-file
selection: **1,016 passed, 1 skipped**, with **86% changed-line coverage**
(80% gate). Full collection found **33,681 tests**. The 90 always-required repository guards,
scoped Black, Ruff and typing passed. CI selection explicitly includes the new
gateway identity/boundary and graph-runtime behavior. Provider construction,
policy mapping and synthetic upstream tests do not establish live subscription
compatibility or entitlement.

No blocking authorization/policy findings remain in the inspected scope. This
review does not certify arbitrary plugins, every deployment topology or production
model quality. Real IdP/provider operations were not repeated during consolidation.

Policy denials are terminal, and configured gateway failure cannot select a
direct upstream fallback. A buffered **non-policy transport failure** may still
be converted to an empty result and trigger a subsequent completion. Sandhi
separately authorizes/meters that request, but inference work may be duplicated.
Do not claim universal exactly-once/no-replay execution. The remote evaluator
no-replay rule is a distinct Sandhi contract.

Deploy matching tested Sandhi artifacts for the new route; the committed default
dependency pin is not a claim that the new gateway server has already shipped.
Credential broker supervision, browser SSO acceptance, shared distributed budgets,
alerts and actual provider/data-routing acceptance remain deployment follow-ups.
