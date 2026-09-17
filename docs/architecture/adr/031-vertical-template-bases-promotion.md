# ADR-031: Vertical Template Bases — extract shared bases into `victor_contracts.verticals.bases`

## Metadata

- **Status**: Accepted
- **Date**: 2026-09-07
- **Decision Makers**: Vijaykumar Singh
- **Related**: FEP-0031 (chat runtime inversion), `victor_contracts/verticals/` (destination,
  prior art), ADR-027 (strategy-fidelity naming discipline)
- **Scope**: The per-vertical template base files in the four small verticals
  (`victor-{devops,research,rag,dataanalysis}`). `victor/contrib/` itself is not moved; its
  base classes are reconciled with the new `victor_contracts.verticals.bases` family-by-family
  (see Context). Vertical registration semantics are unchanged, as is every vertical's
  observable behavior.

## Context

Backlog item 28 ("Contrib bases → `victor_contracts.verticals`; de-template the 4 small
verticals, ~12k LOC") contains two premises that 2026-09-07 verification split apart:

1. **`victor/contrib/` is already centralized — but it is not the template source.** It is
   38 files / 4,923 lines of shared optional runtime packages (codebase/editing/lsp/parsing/
   safety/vectorstores/workflows), and it *does* carry vertical base classes
   (`BaseSafetyExtension`, `VerticalSafetyMixin` in `contrib/safety/`, `BaseModeConfigProvider`
   in `contrib/mode_config/`, `BaseConversationManager` in `contrib/conversation/`) — consumed
   by `victor/benchmark/mode_config.py`, `victor/benchmark/safety.py`, and the service-provider
   seam. U9-F5's own remedy was to promote *these* contrib bases into
   `victor_contracts.verticals`. However, the vertical copies are not derived from them (every
   vertical copy is md5-distinct template growth, unrelated text), so this ADR's extraction
   route stands — with the explicit requirement that the new `victor_contracts.verticals.bases`
   either absorb the corresponding contrib bases or be reconciled with them family-by-family
   during extraction, so contracts does not end up with two parallel base hierarchies for the
   same concern.
2. **The per-vertical template bases are the real problem.** Twelve filename families are
   copy-derived across the four small verticals — `capabilities.py` (3,216 lines summed),
   `handlers.py` (2,053), `escape_hatches.py` (1,883), `safety.py` (1,840),
   `tool_dependencies.py` (1,272), `protocols.py` (1,220), `conversation_enhanced.py` (1,103),
   `safety_enhanced.py` (961), `assistant.py` (955), `prompts.py` (672), `mode_config.py`
   (473), `plugin.py` (192) — roughly 15.8k lines across the copies, out of ~31.1k total
   vertical LOC.

Critically, **every copy is md5-distinct and the diffs are large** (e.g. `capabilities.py`
devops-vs-research diverges by ~1,008 lines on ~790-line files): these are template files with
domain content grown in, not byte duplicates. The "~12k LOC de-templating" figure is therefore
an estimate of *boilerplate share*, not literal duplicate bytes. Success cannot be measured as
dedup-equals; it is measured as **each vertical shrinking to its domain delta**.

The destination already exists with prior art: `victor_contracts/verticals/` ships
`registration.py` (`@register_vertical`), `manifest.py`, `validation.py`,
`tool_dependencies.py`, `protocols/`, and five mixins (`ExtensionProvider`, `PromptMetadata`,
`RL`, `Team`, `WorkflowMetadata`) — the exact shape the extraction extends. The
`feps/README.md` FEP gate ("✅ Vertical capability promotion to framework") applies to promoting
vertical code *up*; extracting shared bases *out of* templated verticals into contracts is the
lighter ADR route this document takes.

## Decision

**Extract the shared skeleton of the twelve template families into
`victor_contracts.verticals.bases` as base classes/hooks; each small vertical shrinks to its
domain delta by subclassing.** Extraction is hook-shaped, not copy-shaped: where vertical copies
differ, the difference becomes either a template method the base calls, a declarative manifest
field, or a registered hook — decided per family during extraction, defaulting to the
`victor_contracts/verticals/mixins/` style.

Rules:

- **No behavior change per family.** Each extracted family lands with before/after behavioral
  equivalence tests on all four verticals (the verticals' contract test suites are the gate).
- **Extraction order by stability**: start with the families whose vertical copies differ least
  (likely `plugin.py`, `mode_config.py`, `prompts.py` — small files), and treat the large
  domain-heavy families (`capabilities.py`, `handlers.py`) last, extracting only their shared
  skeleton.
- **The four small verticals only.** `victor-coding` is not template-derived and stays as-is.
- **`victor/contrib/` is explicitly out of scope** — already centralized; no move, no rename.

## Consequences

- Each small vertical's LOC drops by its boilerplate share (the ~12k estimate is the ceiling;
  the realizable number is whatever the hook extraction yields — measured, not assumed, per
  family).
- Template fixes (a bug in the shared skeleton) land once in `victor_contracts.verticals.bases`
  instead of four times; the md5-distinct drift problem stops growing.
- The contracts package grows by shared *base classes* — the mixins precedent — not by runtime
  subsystem weight (contrast with the FEP-0033 analysis of why 37.8k runtime lines must NOT go
  to contracts).
- Follows ADR-027's naming discipline: extracted bases get honest names (`VerticalToolBase`,
  not `ContribBase`) reflecting where they now live.
- The remaining de-templating half of item 28 (shrinking the four verticals to domain delta)
  proceeds family-by-family after this ADR's bases land, tracked in the README item map.
