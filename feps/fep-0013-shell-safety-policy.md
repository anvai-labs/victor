---
fep: "0013"
title: "Damage-scoped ShellSafetyPolicy — replace the readonly allowlist with a composable, context-aware, RL-seamed policy"
type: Standards Track
status: Draft
created: 2026-06-30
modified: 2026-06-30
authors:
  - name: Vijaykumar Singh
    email: vijay@anvaiops.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/anvai-labs/victor/discussions/0013
---

# FEP-0013: Damage-scoped ShellSafetyPolicy

## Summary

The shell tool's `readonly` mode is simultaneously Victor's most important
safety mechanism and its single largest execution blocker. Today it is a
**binary flag** (`readonly=True/False`) backed by a **hardcoded command
allowlist** (`_validate_readonly_command`, ~330 LOC with eight per-tool
subcommand allowlists in `victor/tools/bash.py`). This FEP replaces it with a
**policy object** — a pluggable `ShellSafetyPolicy` protocol that is
**context-aware**, **damage-scoped** (protects *assets*, not *commands*),
**session-scoped** (configured through `SessionConfig`), and **RL-seamed**
(every decision is logged for later learning).

The central shift is first-principles: the safety invariant is not *"only run
allowlisted commands"* — it is *"prevent escape from the workspace and
irreversible / cross-boundary damage, while allowing every in-scope action."*
`pip install`, `xargs`, `sed -i`, redirects, `git checkout`, and heredocs are
all **safe inside the workspace**; they are only dangerous when they escape it.
The current model cannot express that distinction, so it blocks them
unconditionally and the model rarely passes `readonly=False` to override.

Phase 1 (this FEP's implementation target) lands the seam: the policy object,
the `SessionConfig.shell_safety` wiring, and a damage-scoped default policy —
additive and non-breaking, with the legacy allowlist preserved as a
back-compat policy.

## Motivation

### The current design, and why it blocks real work

`shell()` (`victor/tools/bash.py:1338`) takes `readonly=True` by default.
When `readonly` is true, `_validate_readonly_command()` (`bash.py:725`) parses
the command, splits compounds, scans substitutions, and runs each simple
command through `_validate_single_readonly_command()` (`bash.py:758`) — a
~300-line function that maintains hand-curated readonly-subcommand sets for
`git`, `pip`, `npm`, `cargo`, `go`, `gh`, `az`, `kubectl`, `docker`, `podman`,
`helm`, `yarn`, `pnpm`, `tar`, `unzip`, `ip`, plus special cases for `sed`,
`black`, `python`, and pipe-to-shell. Separately, `is_dangerous_command()`
(`victor/security/command_safety.py`) blocks a small set of catastrophic
patterns (`rm -rf /`, `dd`, `mkfs`, fork bombs, `curl | sh`).

This is **command-scoped, not damage-scoped**: it asks "is this command on the
list?" instead of "what could this command irreversibly damage?"

### Mined failure data (all from this machine, 2026-06-30)

**1. Real rejections in interactive sessions — `~/.victor/logs/victor.log*`**

`grep` for `"not allowed in readonly mode"` and `"Dangerous command blocked"`
across six rotated logs yields **144 actual rejections**. Tabulated by the
blocked command:

| Blocked command | Count | Legitimate for agent work? |
|-----------------|------:|----------------------------|
| `xargs`                 | 32 | ✅ `find … \| xargs grep` / `-exec` |
| redirection `>`         | 30 | ✅ writing a file via shell |
| `sed -i`                | 16 | ✅ in-place edit (common patch flow) |
| `\| sh` / `\| bash`     | 12 | ⚠️ sometimes (installers); blocked wholesale |
| `sqlite3`               |  6 | ✅ DB inspection |
| `git checkout`          |  6 | ✅ branch switching |
| `timeout …`             |  4 | ✅ `timeout 30 pytest …` |
| `rm`                    |  4 | ✅ workspace cleanup |
| `pip install`           |  4 | ✅ dependency setup |
| `curl`                  |  4 | ✅ fetch (non-piped) |
| `terraform`/`pipdeptree`/`pdftotext`/`mv`/`git fetch`/`docker write`/`chmod`/`cd`/`/…/python` | 2 each | ✅ mixed legit |

**~95% of these rejections are legitimate agent actions.** `xargs`, redirects,
and `sed -i` alone account for **78/144 (54%)** — and every one is a command an
agent legitimately needs to explore, edit, and verify code.

**2. Block rate — `~/.victor/logs/usage.jsonl`**

Of **1,801** shell `tool_result` events, **89 (~4.9%)** were
readonly/dangerous blocks. Roughly **one in twenty** shell calls is rejected.

**3. SWE-bench overnight run — `swebench_zai_20260630.json` (glm-5.2 / zai)**

| Metric | Value |
|--------|-------|
| Tasks | 50 |
| Pass rate | **32%** (16 passed / 28 failed / **6 timeouts**) |
| Tool calls | 880 |
| Avg tool calls — passed | 21 |
| Avg tool calls — failed | 13 |
| Avg tool calls — **timeout** | **28** (max 37) |

Two failure signatures dominate and both are *execution-environment* problems,
not reasoning problems:

- **Dependency / harness failure (astropy ×4):** *"Patch applied successfully
  but tests could not run (0 collected). Install project deps or use Docker."*
  The agent produced a correct patch but **could not install the project or run
  its test suite** — exactly the `pip install` / build / test-run class that the
  readonly allowlist blocks outside benchmark mode.
- **Timeouts (×6):** all categorised `execution.timeout.task_timeout`; agents
  burn **17–37 tool calls in only 2–3 turns** thrashing on slow shell/test
  commands. Execution friction, not capability.

**4. The benchmark "fix" is the opposite extreme.** Because the allowlist is
unusable for benchmarks, `agent_adapter._on_tool_start_hook`
(`victor/evaluation/agent_adapter.py`) **mutates `arguments["readonly"]=False`
unconditionally** for benchmark sessions. Confirmed: benchmark traces contain
**zero** readonly-block strings. So the spectrum today is *block everything
legitimate* (interactive) ↔ *allow literally everything* (benchmark). Neither
end is correct; both are symptoms of the missing abstraction.

### Goals

1. **Policy object, not a list.** Open/closed: a new safety rule plugs in
   without editing `bash.py`. Interface segregation: the shell tool depends on a
   `ShellSafetyPolicy` protocol, not a concrete implementation.
2. **Damage-scoped.** Define what is *protected* (workspace-escape targets,
   `~/.victor/`, system dirs, cross-process/network) and allow everything else
   in-scope. The invariant is "no escape, no irreversible damage" — not "no
   `pip`."
3. **Context-aware & composable.** Policy varies by session context —
   `STRICT` (interactive), `BENCHMARK` (permissive inside the task repo),
   `AUTONOMOUS` (medium). A permissive policy substitutes for a strict one
   (Liskov) and composes via a chain (short-circuit on DENY).
4. **Session-scoped configuration.** `SessionConfig` gains `shell_safety`
   (profile, protected paths, workspace root, network policy); the orchestrator
   wires it to the shell tool at construction. Not a tool-decorator default.
5. **RL seam.** Every decision is logged to `decision_log`; outcomes to
   `rl_outcome`. A future `RLShellSafetyPolicy` adapts thresholds from observed
   outcomes — replacing a static list with a learned one.
6. **Co-designed surface.** The safety policy and the execution surface are
   designed together: `readonly`/`action`/`dangerous` become *hints* the policy
   interprets, removing the adapter's bit-flip hack.

### Non-Goals

- **Not** a full shell AST analyser. Path-extraction is best-effort; the
  catastrophic blocklist (L0) remains the hard floor.
- **Not** removing the legacy allowlist in Phase 1. It is preserved as a
  back-compat policy; migration to damage-scoping is incremental (Phase 2+).
- **Not** the RL training loop itself (Phase 3). This FEP defines the *seam*
  (decision logging) only.
- **Not** OS-level sandboxing. `bwrap`/`seatbelt` (FEP sandbox work) is
  orthogonal and complementary; the policy can *request* a sandbox as one of its
  decisions.

## Proposed Change

### The invariant (first principles)

A shell command is **safe to run** iff it does **none** of:

1. **Escape the workspace** — write or affect paths outside the session's
   working tree (and explicitly-allowed auxiliary roots: the venv, pip cache,
   temp).
2. **Corrupt protected assets** — `~/.victor/` (state, DBs, profiles, keys),
   system directories (`/etc`, `/usr`, `/System`, `/boot`), or other projects.
3. **Do irreversible / cross-boundary damage** — `rm -rf /`, `dd of=/dev/`,
   `mkfs`, fork bombs, `curl … | sh`, privilege escalation, secret exfiltration.
4. **Exceed the session's blast radius** — network egress, backgrounded
   processes, or container escapes beyond the declared profile.

Notice what is **absent** from the invariant: the name of the command. `pip
install`, `xargs`, `sed -i`, and `>` violate *none* of these when they operate
inside the workspace. They are safe. The current model rejects them because it
never asks the workspace question.

### Layered safety model

```
                 ┌─────────────────────────────────────────────┐
   command ───▶ │  ShellSafetyPolicy.evaluate(ctx)            │ ──▶ Decision
                 │                                             │     (ALLOW /
                 │  L0  Catastrophic blocklist   (always)      │      DENY /
                 │      rm -rf /, dd, mkfs, fork, curl|sh      │      ASK)
                 │      ── existing command_safety.py ──       │
                 │                                             │
                 │  L1  Workspace containment                  │
                 │      extract write/network targets          │
                 │      resolve vs cwd; reject escape          │
                 │      into protected zones                   │
                 │                                             │
                 │  L2  Context profile  (STRICT/BENCH/AUTO)   │
                 │      network egress, backgrounding,         │
                 │      cross-process scope                    │
                 │                                             │
                 │  L3  RL + ASK  (seam; Phase 3)              │
                 │      learned thresholds, human approval     │
                 └─────────────────────────────────────────────┘
```

- **L0** is the non-negotiable floor: the existing
  `is_dangerous_command()` blocklist, applied in every context. It is small
  (6 exact + 13 substring patterns) and *damage*-focused — already the right
  shape.
- **L1** is the new core. It extracts the command's **write targets** (redirect
  operators, and path arguments to mutating builtins/externals like `rm`,
  `cp`, `mv`, `pip install --target`, `git checkout --`) and **network
  targets** (`curl`/`wget`/`ssh`/`pip install` URLs), resolves them against
  `cwd`, and rejects any that escape the workspace boundary or land in a
  protected zone. Commands with no extractable mutating target default to
  **allow** at this layer.
- **L2** applies the session profile's blast-radius rules (network on/off,
  backgrounding, container scope).
- **L3** is the adaptation seam (decision logging now; learning later).

### Architecture: today vs. proposed

```
 TODAY                                    PROPOSED
 ─────                                    ────────
 SessionConfig (no shell config)          SessionConfig
        │                                       │  shell_safety: ShellSafetyConfig
        ▼                                       ▼
 AgentFactory ──▶ Orchestrator             AgentFactory ──▶ Orchestrator
                       │                                         │  set_shell_safety_policy(ctx)
                       ▼                                         ▼
              ToolRegistry                              ToolRegistry
                       │                                         │
                       ▼                                         ▼
       shell(readonly=True)                     shell() ──▶ get_shell_safety_policy()
              │                                               .evaluate(ctx) ──▶ Decision
              ▼                                                     │
   _validate_readonly_command  ◀── 330-LOC                         ▼
   (hardcoded allowlist)                                  ShellSafetyPolicy (protocol)
                                                              │         │         │
                                                  LegacyAllowlist  DamageScoped  RL(seam)
                                                  (back-compat)    (new)         (Phase 3)

 benchmark path:                                benchmark path:
   adapter mutates readonly=False  ◀── hack       BENCHMARK profile (permissive,
                                                   in-repo)  ◀── first-class
```

### Decision flow

```
 shell(cmd, cwd, readonly=hint, action=hint)
        │
        ▼
 ctx = ShellCommandContext(cmd, cwd, session_context, profile, hints)
        │
        ▼
 decision = policy.evaluate(ctx)
        │
        ├─ ALLOW  ─▶ run; decision.category + risk logged to decision_log
        ├─ ASK    ─▶ route through governance PolicyEngine (FEP-0005) approval
        └─ DENY   ─▶ return structured error citing the violated invariant
                     (NOT "not on the list") + the remediation
        │
        ▼
 outcome ─▶ rl_outcome (success/failure/error) keyed by cmd-signature
```

The DENY message is itself a design change: today it dumps *"Allowed commands:
., [, ag, arxiv, …"* (the allowlist). The new message names the **invariant
violated** — e.g. *"denied: writes outside workspace (`/etc/hosts`); run inside
`<repo>` or pass `cwd=`"* — which is actionable for the model.

## API Changes

### New module: `victor/security/shell_safety_policy.py`

```python
class SafetyVerdict(str, Enum):
    ALLOW = "allow"
    DENY  = "deny"
    ASK   = "ask"

@dataclass(frozen=True)
class ShellCommandContext:
    command: str
    cwd: str
    session_context: str            # "interactive" | "benchmark" | "autonomous"
    profile: "ShellSafetyProfile"
    workspace_root: Optional[str]
    protected_paths: tuple[str, ...]
    allow_network: Optional[bool]   # None = profile default
    readonly_hint: Optional[bool]   # what the model passed
    action_hint: Optional[str]      # read/write/network/exec
    correlation_id: Optional[str]   # for decision_log join

@dataclass(frozen=True)
class ShellSafetyDecision:
    verdict: SafetyVerdict
    effective_readonly: bool        # back-compat with shell()'s existing gate
    reason: str                     # human/actionable
    category: str                   # RL feature key, e.g. "workspace_write"
    risk_score: float               # 0.0–1.0
    invariant: Optional[str]        # which of the 4 clauses, if denied

class ShellSafetyPolicy(Protocol):
    """Evaluate a shell command for the session's safety context."""
    def evaluate(self, ctx: ShellCommandContext) -> ShellSafetyDecision: ...
    @property
    def name(self) -> str: ...

class CompositeShellSafetyPolicy(ShellSafetyPolicy):
    """Chain policies; short-circuit on first DENY, collapse ASK."""
    def __init__(self, policies: Sequence[ShellSafetyPolicy]): ...

class ShellSafetyProfile(str, Enum):
    LEGACY     = "legacy"        # back-compat: reproduce current allowlist
    STRICT     = "strict"        # interactive: workspace-writes ok, network deny
    BENCHMARK  = "benchmark"     # in-repo permissive incl. pip install/build/test
    AUTONOMOUS = "autonomous"    # workspace writes ok, network ASK
    CUSTOM     = "custom"

# Concrete Phase-1 policies:
class LegacyAllowlistPolicy:      # wraps existing _validate_readonly_command
class DamageScopedShellSafetyPolicy:  # L0 + L1 + L2
def policy_for_profile(profile, **cfg) -> ShellSafetyPolicy: ...

# Session-scoped accessor (contextvar so parallel benchmark sessions don't clash):
def get_shell_safety_policy() -> ShellSafetyPolicy: ...
def set_shell_safety_policy(policy) -> None: ...
def reset_shell_safety_policy() -> None: ...   # test isolation
```

### `SessionConfig` + `Settings`

```python
# victor/framework/session_config.py
@dataclass(frozen=True)
class ShellSafetyConfig:
    profile: str = "legacy"                 # safe default = unchanged behavior
    workspace_root: Optional[str] = None    # None = cwd at session start
    protected_paths: tuple[str, ...] = ()   # extra, e.g. ~/.victor always implicit
    allow_network: Optional[bool] = None
    extra_allow_patterns: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()

# added to SessionConfig:
shell_safety: ShellSafetyConfig = field(default_factory=ShellSafetyConfig)
# and ShellSafetyConfig.from_cli(profile=..., allow_network=...) + apply_to_settings()
# writes to settings.shell_safety (new settings group)
```

### Shell tool integration (additive)

`shell()` consults the policy **when one is configured for the session**; with
the default `LEGACY` profile (or no policy set) behaviour is byte-identical to
today:

```python
# in shell(), replacing the hardcoded readonly gate:
_policy = get_shell_safety_policy()
if _policy.name != "legacy-allowlist":        # opt-in only
    decision = _policy.evaluate(ctx)
    if decision.verdict == SafetyVerdict.DENY:
        return _denied_result(decision)       # actionable invariant message
    readonly = decision.effective_readonly
    # ASK routes through the existing governance PolicyEngine (FEP-0005)
# else: existing _validate_readonly_command path (unchanged)
```

### Benchmark adapter migration

`agent_adapter._on_tool_start_hook` stops mutating `readonly` and instead sets
`SessionConfig.shell_safety = ShellSafetyConfig(profile="benchmark",
workspace_root=<task repo>)` at session creation. Permissive-by-policy, scoped
to the repo — not bit-flipped.

## Compatibility

- **Default = no change.** `SessionConfig.shell_safety.profile` defaults to
  `"legacy"`; the legacy allowlist is preserved as `LegacyAllowlistPolicy`. All
  existing tests and interactive behaviour are unchanged until a caller opts in.
- **Back-compat params retained.** `readonly`/`action`/`dangerous` remain on
  `shell()` as *hints* the policy may honour or override. No schema change for
  the model.
- **CI guards.** This is additive framework surface (`victor/security/`); it
  does not touch `archive/`, the SDK boundary, or feature-flag counts. A new
  unit-test module covers the policy; `make test` must stay green.

## Benefits

- **Unblocks the 78/144 top blockers** (`xargs`, `>`, `sed -i`) and the rest of
  the legitimate set — inside the workspace — without disabling safety.
- **Removes the adapter hack** — benchmarks get a first-class permissive
  profile instead of `readonly=False` mutation.
- **Open/closed.** New safety rules, providers, or verticals can ship a
  `ShellSafetyPolicy` without touching `bash.py`.
- **Actionable denials** model can act on (names the invariant + remediation).
- **RL-ready.** Decision logging closes the loop toward a learned, per-context
  policy — the static list's successor.
- **Composes with governance.** ASK verdicts route through the existing
  FEP-0005 `PolicyEngine`, reusing the approval surface.

## Drawbacks and Alternatives

- **Drawback:** path-extraction is imperfect (shell is hard to parse). Mitigated
  by L0 as a hard floor and by treating un-extractable mutating commands as
  context-dependent (STRICT→ASK, BENCHMARK→allow).
- **Drawback:** more moving parts than a list. Justified by the open/closed and
  context-awareness goals and by removing the adapter hack.
- **Alternative considered:** extend the allowlist to cover `xargs`/redirects.
  Rejected — it's whack-a-mole; the mined data shows the blocker set is open-ended
  and the fundamental granularity is wrong.
- **Alternative considered:** implement as a governance `Policy` (FEP-0005)
  directly. Rejected for the *primary* path: the governance engine operates at
  tool-call granularity (allow/deny the whole call) and does not understand
  shell command structure. A thin **adapter** can expose `ShellSafetyPolicy`
  decisions *to* the governance engine for ASK routing (Phase 2), getting the
  best of both without forcing shell semantics into the generic engine.

## Unresolved Questions

1. Should `BENCHMARK` profile auto-derive `workspace_root` from the task repo,
   or require it explicit? (Lean: auto-derive, allow override.)
2. Path-resolution for symlinks that cross the workspace boundary — follow or
   deny? (Lean: deny by default, configurable.)
3. Phase-3 RL feature shape: cmd-signature hash vs. structural features (has
   redirect / has network / target zone). Deferred to a follow-up FEP once the
   decision log has data.

## Implementation Plan

| Phase | Scope | Status |
|-------|-------|--------|
| **1** (this PR) | `ShellSafetyPolicy` protocol + context/decision types; `DamageScopedShellSafetyPolicy` (L0+L1+L2); `LegacyAllowlistPolicy` wrapper; `CompositeShellSafetyPolicy`; `ShellSafetyProfile`; contextvar accessor; `SessionConfig.shell_safety` + `apply_to_settings`; shell tool opt-in consult; unit tests; **non-breaking default**. | Proposed |
| 2 | Migrate benchmark adapter to `BENCHMARK` profile (remove `readonly=False` hack); flip interactive default to `STRICT`; governance `Policy` adapter for ASK; integration tests against the 144-block corpus. | Next |
| 3 | `decision_log` write on every `evaluate()`; `rl_outcome` join; `RLShellSafetyPolicy` reading prior outcomes; A/B harness (`shell_safety_ab.py`) mirroring the completion/temperature harnesses. | Future |
| 4 | Retire `LegacyAllowlistPolicy`; deprecate the inline `_validate_readonly_command` allowlist in `bash.py`. | Future |

### Phase 1 acceptance criteria

- `victor/security/shell_safety_policy.py` ships the protocol, types,
  `DamageScopedShellSafetyPolicy`, `LegacyAllowlistPolicy`,
  `CompositeShellSafetyPolicy`, `ShellSafetyProfile`, and the contextvar
  accessor.
- `SessionConfig` gains `ShellSafetyConfig` + `from_cli`/`apply_to_settings`.
- `shell()` consults the policy **only when opted in**; default session is
  behaviour-identical (proven by a parity test).
- New unit tests prove: (a) the 144-blocked-command corpus is **ALLOW** under
  `BENCHMARK`/`STRICT` when in-workspace; (b) `rm -rf /`, `dd`, `curl|sh`,
  writes to `/etc`/`~/.victor/` are **DENY** in every profile; (c) ASK routing
  for network under `AUTONOMOUS`.
- `make test`, `make lint` green; FEP passes `victor fep validate`.

## Migration Path

1. Land Phase 1 additive (default `LEGACY`). Zero behaviour change.
2. Opt-in: benchmarks set `profile="benchmark"`; pilot interactive sessions set
   `profile="strict"`. Measure block-rate and task success (reuse the SWE-bench
   harness + `completion/temperature` A/B infra).
3. On parity/green, flip defaults (Phase 2), then log decisions (Phase 3), then
   retire the legacy list (Phase 4).

## References

- `victor/tools/bash.py:725` — `_validate_readonly_command` (the allowlist)
- `victor/tools/bash.py:1338` — `shell()` signature
- `victor/security/command_safety.py` — catastrophic blocklist (L0 model)
- `victor/evaluation/agent_adapter.py` — `_on_tool_start_hook` (the hack)
- `victor/framework/session_config.py` — `SessionConfig` sub-config pattern
- `victor/framework/policies/` — FEP-0005 governance `Policy` engine (ASK route)
- Mined data: `~/.victor/logs/victor.log*`, `~/.victor/logs/usage.jsonl`,
  `~/.victor/evaluations/swebench_zai_20260630.json`

## Review Process

Standard 14-day Framework Enhancement Proposal review. Validation:
`victor fep validate feps/fep-0013-shell-safety-policy.md` and the
`fep-validation.yml` CI job (front-matter schema + numbering consistency).

## Acceptance Criteria

- FEP-0013 validates and merges after review.
- Phase 1 PR lands additive and non-breaking; `make test` + `make lint` green.
- Decision object and `SessionConfig` seam are in place for Phase 2/3.
