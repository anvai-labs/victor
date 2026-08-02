# EVR-2 Human-Label Validation Protocol (ADR-011)

**Status**: Pre-registered 2026-08-02, BEFORE any labels were collected.
**Rule**: these thresholds and procedures may not change once labeling begins;
a run that motivates changing them is a failed run, recorded as such in
[FINDINGS](../../benchmarks/judge_calibration/FINDINGS.md).

## Why

FINDINGS runs 1–11 measured judges against **programmatic verifier gold**
(workspace-state checks). ADR-011 and the
[flag-graduation policy](flag-graduation-policy.md) require validation against
**human labels** before any judge-gated completion strategy becomes a default.
This protocol validates two things at once: the judge (does it agree with
humans?) and the verifier gold itself (did runs 1–11 measure against labels a
human would endorse?).

## Label set

~150–200 items, three strata:

1. The 48 real-agent trajectories from the run-11 distribution (re-exported).
2. One fresh 96-trajectory real-agent run (`--agent-profile`, `--variants 16`,
   two-phase).
3. ~20–40 SWE-bench-lite smoke trajectories (cross-corpus external validity),
   once available from the evidence workstream.

Unit of labeling: **binary completion verdict per trajectory** (matches the
harness scope note: gold labels are binary completion verdicts only).

## Annotators and blinding

- **Primary**: the maintainer, labeling from the exported labeling pack
  (`--export-labeling-pack`) — the pack is the judge's blinded view (prompt +
  transcript + workspace snapshot) and structurally contains no verifier
  verdict or judge score.
- **Secondary**: one LLM annotator, **disjoint from every judge candidate and
  from the agent under test**, labeling blind from the same pack.
- Disagreements between primary and secondary get a maintainer re-audit with
  written rationale; the audited maintainer label is the final human gold.
- Labels are committed under `benchmarks/judge_calibration/labels/` as JSONL:
  `{"task_id", "label", "annotator", "rationale"}`.

## Pre-registered thresholds

Computed by the overlay (`--human-labels`, module
`victor/evaluation/human_label_overlay.py`):

| Check | Threshold | On failure |
|---|---|---|
| human↔verifier Cohen's κ | **≥ 0.8** | **STOP THE LINE** — verifier gold is invalid; FINDINGS runs 1–11 conclusions are void; fix corpus verifiers, re-validate, re-run |
| human↔judge Krippendorff α (overall) | **≥ 0.7** | Judge not graduated; rubric stays opt-in |
| human↔judge α (per family) | **≥ 0.7** for every family with n ≥ 16; families below n = 16 are directional only and not claimed | Judge not graduated |
| human↔secondary κ | reported, never gating | Audit disagreements; note in FINDINGS |

VOID conditions (inherited from the harness integrity guards): any grading-call
failures or ungradable outputs in the underlying run; single-class gold
(α cannot measure discrimination); unlabeled template lines in the labels file
(the loader rejects partial label sets).

## Consequences

- **All checks pass** → EVR-2 is complete for the measured judge identities;
  the rubric default flip (eval-loop program PR-8) is unblocked once EVR-3
  parity also holds. Judge identities stay pinned via
  `agent.rubric_judge_calibrated_models` (ADR-011 pinning gate).
- **Verifier κ fails** → stop-the-line remediation before anything else.
- **Judge α fails** → the judge stays untrusted; the honest result ships in
  FINDINGS either way.

## Acknowledged limitations

n=1 human annotator (single-maintainer project), mitigated by the blind
secondary annotator and audited disagreements — stated openly wherever these
results are published, not hidden.
