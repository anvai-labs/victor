---
fep: "0024"
title: "Pluggable code-correction — entry-point validators, MiddlewareChain routing, argument-kind trait"
type: Standards Track
status: Draft
created: 2026-07-22
modified: 2026-07-22
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions
---

# FEP-0024: Pluggable code-correction — entry-point validators, MiddlewareChain routing, argument-kind trait

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Proposed Change](#proposed-change)
4. [Benefits](#benefits)
5. [Drawbacks and Alternatives](#drawbacks-and-alternatives)
6. [Unresolved Questions](#unresolved-questions)
7. [Implementation Plan](#implementation-plan)
8. [Migration Path](#migration-path)
9. [Compatibility](#compatibility)
10. [References](#references)
11. [Review Process](#review-process)
12. [Acceptance Criteria](#acceptance-criteria)

---

## Summary

Victor's code-correction subsystem (validate + auto-fix tool arguments before
execution) currently has three structural weaknesses, exposed by the
`sandhi-c3966e22` incident (a markdown document written via `write` was silently
truncated to one code block by the corrector, looping the agent — fixed in #622 by
gating correction on the tool's `access_mode` trait and making `clean_markdown`
non-destructive):

1. **Validator discovery is path-scanned from one fixed directory.** Adding a
   language requires editing four hardcoded points (`Language` enum, a validator
   file under `victor/evaluation/correction/validators/`, `detector` maps, and
   `MARKDOWN_LANGUAGE_MARKERS`/`LANGUAGE_MARKERS`). External verticals and future
   **LSP-backed** validators cannot contribute without modifying `victor/`.
2. **The correction transform bypasses the pluggable `MiddlewareChain`.** It is
   inlined (and, until #622, duplicated) inside `tool_pipeline.py` /
   `tool_executor.py`, so it is not a first-class, extensible argument transform.
3. **There is no per-argument semantic kind.** #622 gates on the *tool's*
   `access_mode` (executable-code vs file-authoring) as a proxy; a tool like a
   notebook runner that both executes cells *and* writes files cannot express
   "this argument is executable code, that one is file content."

This FEP proposes three coordinated, **additive** changes: (a) an **entry-point
validator registry** (`victor.code_validators`) so languages/validators register
pluggably; (b) routing correction through the **existing `MiddlewareChain`** so it
is one extensible transform among many; and (c) a **per-argument kind trait** in
the FEP-0009 `ToolContract`. All are backward compatible; external packages opt in.

## Motivation

### Problem Statement

The correction subsystem is brittle to extend and hard to audit:

- **Adding a language is multi-point and package-internal.** A contributor must
  touch `victor/evaluation/correction/types.py` (`Language` enum), drop a file in
  `victor/evaluation/correction/validators/` (auto-discovered only by `pkgutil`
  scan of that one directory, `registry.py:135-189`), and usually extend
  `detector.py` (`EXTENSION_MAP`, `LANGUAGE_SIGNATURES`) and the markdown-marker
  sets in `base.py`/`feedback.py`. An external vertical (e.g. `victor-coding`
  adding an LSP-backed validator, or `victor-dataanalysis` adding SQL) cannot do
  any of this without editing `victor/` — a boundary violation per the SDK
  contract.
- **Correction is not a pluggable transform.** A genuine, priority-sorted
  `MiddlewareChain` already exists and is wired into the pipeline
  (`victor/agent/middleware_chain.py`; `process_before` at
  `tool_pipeline.py:3057`). But code correction (and shell-policy,
  search-routing, path-redirect) **bypass** it as inline method calls. The
  correction block was even **duplicated** across two execution sites
  (`tool_pipeline.py` + `tool_executor.py`) — #622 collapsed it to one
  `CodeCorrectionMiddleware.process(...)` call, but it is still inline, not a
  chain member. Adding or reordering argument transforms means editing the
  hardcoded sequence.
- **Gating is per-tool, not per-argument.** #622 introduced the correct boundary
  (correct only executable-code tools, by `access_mode == EXECUTE`). But the
  cleanest expression of "is *this argument* executable code?" is a per-argument
  trait, not a whole-tool classification — the tool-level trait is a necessary
  proxy today because `ToolContract` (FEP-0009) carries no argument-level
  metadata.
- **Auditability gap.** Until #622, mutations were logged as "0 issues fixed"
  while content was actually rewritten, and the `code_corrected` /
  `code_validation_errors` flags on `ToolCallResult` were computed but never
  consumed. Making correction a chain member with a structured transform result
  gives one auditable place for all argument mutations.

### Goals

1. Let any package (built-in or external vertical) register a language validator
   via a **Python entry point**, with no edits to `victor/`.
2. Make code correction a **first-class `MiddlewareChain` member**, removing the
   inline/duplicated transform and giving verticals one interception point.
3. Introduce a **per-argument kind** (`executable_code` | `file_content` | `data`)
   on the FEP-0009 `ToolContract`, so the executable-vs-file gate (and future
   per-argument behaviors) is declarative and precise.
4. Preserve the #622 guarantees: file content is never auto-mutated; the gate is
   semantic and fail-safe.

### Non-Goals

- Not changing the *corrector algorithms* themselves (syntax/import validation,
  `fix()`). Only how validators are discovered and how the transform is wired.
- Not removing `SelfCorrector` / `CodeCorrectionMiddleware` — they become the
  chain member + registry consumers.
- Not re-broadening the gate to file content (the #622 boundary is retained).

## Proposed Change

### High-Level Design

```
                                  argument transform pipeline
 tool call ──► MiddlewareChain.process_before(tool, args)   ◀── pluggable, priority-sorted
                 ├─ CodeCorrectionMiddleware  (THIS FEP: a real chain member)
                 │     ├─ should_correct?  ←  per-ArgumentKind (FEP-0009 ext)  + access_mode fallback
                 │     └─ SelfCorrector ──► CodeValidatorRegistry
                 │                          ▲
                 │           victor.code_validators entry points (THIS FEP)
                 │           + builtin path-scan fallback
                 ├─ ShellPolicy, SearchRouting, PathRedirect  (future: also chain members)
                 └─ …
```

### 1. Entry-point validator registry

**New entry-point group** `victor.code_validators`: each entry point resolves to a
`BaseCodeValidator` subclass (or a small factory returning one). `CodeValidatorRegistry`
gains:

```python
def discover_validators(self) -> None:
    # 1. Entry points first (external packages, LSP-backed validators, verticals)
    for ep in importlib.metadata.entry_points(group="victor.code_validators"):
        try:
            validator = ep.load()()   # instantiate
            self.register(validator)
        except Exception as e:
            logger.warning("Failed to load code validator %s: %s", ep.name, e)
    # 2. Built-in path-scan fallback (unchanged, for back-compat)
    ...  # existing pkgutil scan of victor.evaluation.correction.validators
```

**`Language` becomes open**: rather than a closed `Enum` that must be edited per
language, validators declare the language(s) they handle (already via
`supported_languages`); `Language` gains a registry-backed `UNKNOWN`-defaulting
lookup so an externally-registered language resolves without an enum edit. (A
string-keyed alias layer preserves existing `Language.PYTHON` usages.)

`victor-coding`'s future LSP-backed validators (pyright/rust-analyzer/tsserver)
register here instead of living under `victor/`.

### 2. Route correction through `MiddlewareChain`

> **Revision (v1.1, 2026-07-22):** Investigation while implementing Phase 1/3
> **invalidated this section's original premise.** The factory builds
> `ToolPipelineConfig` without `enable_code_correction`, so it defaults to `False`
> — the pipeline's inline correction block is **dead code in production**.
> `ToolExecutor` (`enable_code_correction=True` from settings) is the *actual*
> correction choke point, and it is invoked **directly** by `tool_service`,
> `orchestrator_protocol_adapter`, `parallel_executor`, etc. — not only via the
> pipeline. Therefore routing correction through the *pipeline-level*
> `MiddlewareChain` (the original design) would neither cover all call paths nor
> replace anything live. The original text is retained below for context; the
> **corrected design** follows it.

*Original (pre-finding) design:* `CodeCorrectionMiddleware` implements
`MiddlewareProtocol` (`victor/core/verticals/protocols.py`) and is registered on
the chain. Its `process_before(tool, args)` is the #622 `process(...)` logic. The
inline correction blocks in `tool_pipeline.py` and `tool_executor.py` are
**removed**; the chain already invokes `process_before` at
`tool_pipeline.py:3057`. The `code_corrected` / `code_validation_errors` flags
flow back via the chain's `MiddlewareResult`.

**Corrected design (replaces the above):**

1. **Consolidate to the choke point — the executor — not the pipeline.** Remove
   the dead pipeline inline correction block (it never fires:
   `ToolPipelineConfig.enable_code_correction` defaults `False`). Keep the
   `ToolExecutor` correction (the single active site, covering both
   pipeline-routed and direct calls). This removes the latent double-application
   risk and the deduplication obligation, with **no production behavior change**.
2. **Make `CodeCorrectionMiddleware` implement `MiddlewareProtocol`** (add
   `before_tool_call`/`get_priority`/`get_applicable_tools`) so verticals *can*
   register it on their own `MiddlewareChain` for domain interception. Because
   `MiddlewareProtocol.before_tool_call(tool_name, arguments)` carries no tool
   object — and the gate needs one (`access_mode`/`argument_kinds`) — inject a
   `tool_resolver: Callable[[str], Any]` at construction (the pipeline/executor
   wires `lambda name: executor.get_tool_function(name)`); no `MiddlewareProtocol`
   signature change, so existing vertical middleware is unaffected.
3. **(Optional, larger)** give `ToolExecutor` an optional `middleware_chain` so
   chain-based interception happens at the true choke point for *all* paths.
   Deferred — only needed if verticals must intercept direct-executor calls.

This keeps correction covering every call path (executor is authoritative),
removes dead code, and adds the chain-membership surface for extensibility —
achieving the section's goal without the invalidated pipeline-routing premise.

### 3. Per-argument kind trait (FEP-0009 extension)

Extend `victor_contracts.tools.ToolContract` with an optional per-argument kind
map:

```python
class ArgumentKind(str, Enum):
    EXECUTABLE_CODE = "executable_code"   # runs immediately; eligible for auto-fix
    FILE_CONTENT    = "file_content"      # authored document; validate-only, never mutate
    DATA            = "data"              # opaque payload; no correction

@dataclass(frozen=True)
class ToolContract:
    ...  # existing fields
    argument_kinds: Mapping[str, ArgumentKind] = ()   # param-name -> kind
```

`CodeCorrectionMiddleware.should_correct` then prefers **per-argument** kind when
declared (an argument is correctable iff its kind is `EXECUTABLE_CODE`), falling
back to the tool-level `access_mode == EXECUTE` proxy from #622 when undeclared.
This lets a mixed tool (e.g. a notebook runner) mark `code` as executable and
`content` as file content precisely.

`ToolMetadata.from_contract` bridges the new field; `resolve_contract` precedence
is unchanged.

### API Changes

```python
# External validator (victor-coding or any package) — pyproject.toml:
[project.entry-points."victor.code_validators"]
pyright = "victor_coding.correction.pyright_validator:PyrightValidator"

# Tool author declares an executable-code argument:
class RunNotebook(BaseTool):
    contract = ToolContract(
        access_mode=AccessMode.EXECUTE,
        argument_kinds={"code": ArgumentKind.EXECUTABLE_CODE,
                        "path": ArgumentKind.FILE_CONTENT},
    )
```

No existing tool signatures change. `ToolContract` gains one field; `from_contract`
maps it.

### Dependencies

- `victor-contracts` additive patch bump for the `argument_kinds` field
  (FEP-0009 contract version bump, e.g. `0.7.x → 0.7.(x+1)`); `victor-ai` keeps
  the broad `victor-contracts>=0.6.0,<1.0` range (already admits it), gated by the
  `CapabilityContract` version.
- No new third-party dependencies (entry points are stdlib
  `importlib.metadata`).

## Benefits

- **Extensibility:** add a language/validator with one entry point, zero edits to
  `victor/`. Future LSP-backed validators (the FEP-0019 LSP chain) register
  identically.
- **One transform pipeline:** correction, shell-policy, search-routing,
  path-redirect all become chain members — one place to add/reorder/audit
  argument transforms; verticals get a single interception point.
- **Precise gating:** per-argument kind is the correct semantic unit; mixed tools
  are handled exactly instead of by a tool-level proxy.
- **Auditability:** a chain-member transform yields a structured `MiddlewareResult`
  consumed once; no more dead `code_corrected` flags or misleading "0 issues fixed"
  logs.
- **Decoupled evolution:** the validator vocabulary evolves via entry points
  (packages) and the `CapabilityContract`, independent of `victor-ai` releases.

## Drawbacks and Alternatives

### Drawbacks

- **`Language` open-ness** is a small conceptual shift (closed enum →
  registry-backed). Mitigation: keep the enum names as well-known aliases; the
  registry is additive.
- **Chain-routing refactor** touches the hot tool-execution path. Mitigation: the
  #622 `process()` already isolated the logic; moving it behind `process_before`
  is mechanical, with the existing parity/loop tests as the safety net.
- **`argument_kinds` on `ToolContract`** is a (additive) SDK field. Mitigation:
  optional, defaults to empty; the `access_mode` proxy remains the fallback.

### Alternatives Considered

1. **Status quo (#622 only).** Pros: ship-ready. Cons: external verticals still
   can't add validators; correction still bypasses the chain; gating stays
   per-tool. Rejected for an ecosystem that wants pluggable, LSP-backed correction.
2. **Entry-point registry only (skip chain routing + arg-kind).** Pros: smallest.
   Cons: leaves the duplicated/bypassed transform and per-tool gate in place.
   Rejected: the three pieces are mutually reinforcing.
3. **Move the whole corrector into `victor-contracts`.** Rejected: correction is
   runtime behavior, not a data contract; only the *trait vocabulary*
   (`ArgumentKind`) belongs in the SDK, per FEP-0009's "declarable intent only".

## Unresolved Questions

- **Q1: entry-point instantiation contract** — entry point loads a class vs. a
  factory vs. a module exposing `VALIDATORS`. *Recommended:* a class (consistent
  with the existing `BaseCodeValidator` instantiation in `discover_validators`).
- **Q2: validator precedence when multiple register for one language** (e.g.
  built-in AST validator + an LSP-backed one). *Recommended:* LSP-backed wins when
  available (richer diagnostics), with a `priority`/`preferred` attribute to break
  ties; fall back to built-in otherwise.
- **Q3: should shell-policy/search-routing/path-redirect also migrate to the chain
  in this FEP, or separately?** *Recommended:* separately (this FEP scopes
  correction as the reference migration; the others follow the same pattern).

## Implementation Plan

### Phase 1 — Entry-point validator registry (1 PR, root repo)

- [ ] `CodeValidatorRegistry.discover_validators` loads `victor.code_validators`
      entry points before the path-scan fallback.
- [ ] Registry-backed `Language` resolution (alias layer for the closed enum).
- [ ] Migrate one built-in validator (e.g. `python_validator`) to also declare via
      entry point as the reference; keep path-scan for the rest.
- [ ] Tests: external-validator fixture registers via entry point and is discovered.

**Deliverable:** validators register without editing `victor/`.

### Phase 2 — Consolidate to the executor + chain-membership surface (1 PR, root repo)

> Revised per the v1.1 finding (see §2): the executor, not the pipeline, is the
> correction choke point; the pipeline block is dead code.

- [ ] Remove the dead inline correction block from `tool_pipeline.py` (production
      `enable_code_correction` defaults `False`; executor remains the single active
      site). Update `tests/unit/tools/test_search_router_pipeline.py`
      (`TestToolPipelineCodeCorrection`) accordingly.
- [ ] `CodeCorrectionMiddleware` implements `MiddlewareProtocol`
      (`before_tool_call`/`get_priority`/`get_applicable_tools`) with an injected
      `tool_resolver` so the gate can fetch the tool by name without a protocol
      change. Verticals may register it on their own `MiddlewareChain`.
- [ ] Parity tests: a code-execution tool is corrected identically and a file tool
      untouched, via the executor (the #622 regression suite still passes).

**Deliverable:** single correction site (executor) covering all call paths; no dead
code / no double-application; chain-membership surface for vertical extensibility.

### Phase 3 — Argument-kind trait (1 PR contracts, 1 PR root)

- [ ] `victor_contracts.tools.ArgumentKind` + `ToolContract.argument_kinds`; bump
      the `CapabilityContract` version.
- [ ] `ToolMetadata.from_contract` bridge; `should_correct` prefers per-argument
      kind, falls back to `access_mode`.
- [ ] Parity guard: an undeclared tool resolves identically to #622.

**Deliverable:** per-argument executable-vs-file gating.

### Testing Strategy

- Unit: entry-point discovery; chain-member wiring; `from_contract` mapping for
  `argument_kinds`.
- Integration: an external-validator fixture flows through correction end-to-end;
  a mixed tool's `code` arg is corrected while its `content` arg is not.
- Backward compatibility: tools with no `argument_kinds` and no `contract` behave
  exactly as in #622 (byte-stable gate).

### Rollout Plan

- No feature flag for the registry/chain routing (additive; behavior-preserving
  with the #622 regressions as the gate).
- `argument_kinds` is opt-in (absence = today's behavior).
- Each phase is an independent PR behind this FEP's 14-day review window.

## Migration Path

- Built-in validators: no change required (path-scan fallback remains); may
  optionally declare entry points.
- External verticals: adopt entry points on their own schedule.
- Tool authors: optionally add `argument_kinds`; the `access_mode` proxy continues
  to work.

### Deprecation Timeline

- This FEP: no deprecations. The inline correction call sites are removed in
  Phase 2 (internal; behavior preserved).

## Compatibility

### Backward Compatibility

- **Breaking change:** No.
- **Migration required:** No (opt-in at every layer).
- **Deprecation period:** N/A.

### Version Compatibility

- Minimum Python: 3.10 (unchanged; `importlib.metadata.entry_points` stable).
- `victor-contracts`: additive patch for `argument_kinds` (Phase 3 only).
- `victor-ai`: dependency range unchanged; feature floor enforced by the
  `CapabilityContract` version.

### Vertical Compatibility

- Built-in verticals: no change required.
- External verticals: gain entry-point validator registration + precise arg-kind
  declaration. Lockstep handled by the existing external-vertical compatibility
  workflow.

## References

- PR #622 — `fix(code-correction): stop mutating file content; gate on access_mode
  trait` (the immediate fix; introduces the `process()` seam and contract gate this
  FEP builds on).
- [FEP-0009](./fep-0009-sdk-tool-contract.md) — SDK `ToolContract` (this FEP extends
  it with `argument_kinds`).
- [FEP-0019](./fep-0019-lsp-integrated-verification.md) — LSP integration chain (future
  LSP-backed validators register via the entry-point group proposed here).
- `victor/agent/code_correction_middleware.py`, `victor/evaluation/correction/`
  (`registry.py`, `base.py`, `orchestrator.py`), `victor/agent/middleware_chain.py`,
  `victor/tools/filesystem.py`.
- `CLAUDE.md` — "Extension System & SDK Contract", "Plugin → Vertical → Extension".

## Review Process

### Submission

- **Submitted by:** Vijaykumar Singh
- **Date:** 2026-07-22
- **Pull Request:** (this PR — stacks on #622)

### Review Timeline

- **Initial review period:** 14 days minimum
- **Reviewers assigned:** TBD
- **Discussion thread:** TBD

### Review Checklist

#### Technical Review

- [ ] Entry-point contract is clear (class vs factory — Q1)
- [ ] Validator precedence rules are specified (Q2)
- [ ] Chain routing preserves #622 behavior (parity tests)
- [ ] `argument_kinds` is additive and versioned via `CapabilityContract`
- [ ] No SDK boundary violations (external packages import only `victor_contracts`)

#### Community Review

- [ ] Use cases (LSP-backed validators, SQL/notebook verticals) are understood
- [ ] Migration path is opt-in and clear
- [ ] Alternatives were considered

### Decisions

- **Recommendation:** TBD
- **Decision date:** TBD
- **Approved by:** TBD
- **Rationale:** TBD

### Revision History

1. **v1.0** (2026-07-22): Initial submission. Stacks on #622; proposes entry-point
   validator registry, `MiddlewareChain` routing, and `ToolContract.argument_kinds`.
2. **v1.1** (2026-07-22): Phase 1 (#624) and Phase 3 (#625) implemented (TDD). Phase 2
   **revised after investigation**: the pipeline inline correction block is dead code
   (`ToolPipelineConfig.enable_code_correction` defaults `False`) and `ToolExecutor` is
   the real choke point (called directly by `tool_service`/`orchestrator_protocol_adapter`/
   `parallel_executor`), so pipeline-level `MiddlewareChain` routing was the wrong
   premise. Corrected Phase 2 to: remove the dead pipeline block (executor stays the
   single active site) + make `CodeCorrectionMiddleware` a `MiddlewareProtocol` member
   with an injected `tool_resolver` for vertical extensibility. See §2 and the Phase 2
   plan.

## Acceptance Criteria

### Must-Have Criteria

1. **Entry-point discovery:** a validator declared via `victor.code_validators` in
   an external package is discovered and used with no `victor/` edits.
   - Verification: an external-validator fixture test.
2. **Behavior-preserving chain routing:** a code-execution tool is corrected and a
   file tool is untouched, identical to #622.
   - Verification: #622 regression suite passes unchanged under chain routing.
3. **Additive SDK field:** `argument_kinds` maps cleanly via `from_contract`;
   undeclared tools resolve identically to today.
   - Verification: parity guard test.

### Should-Have Criteria

1. **Reference LSP-backed validator** registered via entry point in `victor-coding`.
   - Priority: Medium.
2. **Validator precedence** (LSP-preferred-over-builtin) documented and tested.
   - Priority: Medium.

### Implementation Requirements

- [ ] Code implementation following the specification (three phases)
- [ ] Comprehensive test coverage (>80% for new code)
- [ ] API/SDK docs updated
- [ ] Changelog entries added
- [ ] Backward compatibility verified (parity with #622)
- [ ] External-vertical compatibility workflow green

### Validation Process

1. **Automated:** entry-point + parity + backward-compat tests in CI.
2. **Manual:** API review against SDK conventions.
3. **Community:** external-vertical maintainers trial entry-point registration.
4. **Final approval:** maintainer sign-off after the 14-day window.

### Success Metrics

- Validators added outside `victor/`: target **≥1** (an LSP-backed validator in
  `victor-coding`).
- Argument transforms running through `MiddlewareChain`: correction as the
  reference; shell-policy/search-routing/path-redirect migrated in follow-ups.

---

## Copyright

This FEP is licensed under the Apache License 2.0, same as the Victor project.
