# ADR-035: Fail-closed sandbox posture by default

## Metadata

- **Status**: Proposed
- **Date**: 2026-11-16
- **Decision Makers**: Vijaykumar Singh
- **Related Documents**: ADR-010 (flag-graduation pattern), FEP-0047 (FOU program), `victor/config/sandbox_settings.py`, `victor/tools/sandbox/backends.py`, `victor/config/security_settings.py`

## Context

Victor's OS-level process sandbox (`victor/tools/sandbox/backends.py`) ships two real
backends — `BubblewrapSandbox` (Linux bwrap mount namespaces) and `SeatbeltSandbox` (macOS
sandbox-exec SBPL) — plus `NoneSandbox`. The posture selection
(`effective_sandbox_policy()` / `resolve_sandbox_backend()`) defaults to **disabled-or-
fail-open**: with no explicit `sandbox_policy`, shell execution runs unsandboxed; under
`preferred`, an unavailable backend degrades to `NoneSandbox` with a one-time warning. The
backends are also self-described as MVP: bwrap omits seccomp and network-egress hardening,
and the Seatbelt profile uses an `(allow default)` base with narrow denies. Meanwhile the
container `CodeSandbox` (victor-coding vertical) already enforces the quality bar Victor
advertises — network disabled by default, memory cap, idle-TTL expiry, fail-closed when
availability is required — but only for vertical users.

The competitive consequence is concrete: OpenAI Codex made default sandboxing its headline
safety property in 2025–2026, and "rich policy machinery that is off by default" reads, to a
new user, identically to "no sandbox". Victor's base `victor chat` shell execution is
unsandboxed today unless a user discovers and opts into a policy whose `preferred` mode can
still silently degrade. This is the worst of both worlds: complexity carried, safety not
delivered.

## Decision

1. **Default posture is `required`, not `preferred` and not disabled.** For new sessions with
   no explicit policy configured, shell-command execution is wrapped by an available sandbox
   or refused. `resolve_sandbox_backend()` changes its no-policy derivation from
   disabled/fail-open to: try bwrap (Linux) → Seatbelt (macOS) → refuse execution with an
   actionable error naming the escape hatch.
2. **The escape hatch is explicit and loud**: `sandbox_policy: "disabled"` in settings or
   `VICTOR_SANDBOX_POLICY=disabled` in the environment restores unsandboxed execution; the
   CLI surfaces `--dangerously-unsandboxed` as the per-invocation form. Every unsandboxed
   execution under an explicit opt-out logs a one-time session warning.
3. **`preferred` keeps its fail-open semantics** for users who choose it explicitly — it
   becomes an opt-in convenience mode, not a default.
4. **Backends are hardened to make the default deliverable**:
   - `BubblewrapSandbox` adds a seccomp filter and an unshare-network egress deny by default
     (network allowlisting via `sandbox_network_isolation`/`sandbox_allowed_mounts` settings,
     which stop being dead knobs).
   - `SeatbeltSandbox` replaces the `(allow default)` base with a deny-default allowlist
     profile covering the workspace write scope; unmatched operations deny rather than allow.
5. **Core promotion of the container bar (FOU-S2)**: the `CodeSandbox` lifecycle quality bar
   (lazy acquire, self-expiry, network-disabled default, resource caps) moves into core
   behind the capability-loader seam (`victor/agent/service_provider.py`) so Docker-available
   installs get container isolation for code execution without the vertical; the OS sandbox
   remains the fallback for installs without Docker.

## Rationale

A sandbox that is off by default protects no one and still costs maintenance; a sandbox that
silently degrades teaches users to ignore warnings. Fail-closed with an explicit, documented
escape is the posture both the security literature and the 2025–2026 competitive baseline
converged on. Delivering it required fixing the backends' MVP gaps — a fail-closed default
over an `(allow default)` Seatbelt profile would only relocate the false sense of security.
The layered model (container when Docker exists, OS namespace sandbox otherwise, refusal
otherwise, explicit opt-out everywhere) keeps air-gapped and minimal installs working with
`disabled` while making *safe* the path of least resistance.

## Consequences

- **Positive**: base shell execution is isolated by default; the advertised safety story
  becomes the observed behavior; existing settings knobs (`sandbox_network_isolation`,
  `sandbox_allowed_mounts`, `sandbox_filesystem_mode`) become load-bearing; the Codex-class
  branding gap closes without new machinery.
- **Negative**: some environments cannot use bwrap/Seatbelt (restricted containers, WSL
  edge cases, CI without user namespaces) and will now **fail to execute shell commands**
  until they set the opt-out — a real behavior change requiring release-note prominence and
  a precise error message naming the exact knob. Hardening Seatbelt to deny-default risks
  breaking legitimate workflows; the allowlist needs a qualification pass against the test
  battery before the default flips.
- **Neutral**: `preferred` mode semantics, the Docker code-executor defaults
  (`code_executor_network_disabled=True` already), and API-key auth gates are unchanged.

## Implementation

FOU Tier-S increments (tracked in `docs/architecture/victor-foundations-plan.md`):

1. **FOU-S1a**: bwrap seccomp + egress deny; Seatbelt deny-default allowlist profile; parity
   and qualification suites extended (RED: egress attempt succeeds today under bwrap; GREEN:
   denied).
2. **FOU-S1b**: posture flip — `effective_sandbox_policy()` derivation to `required`, escape
   hatch + one-time opt-out warning, `/system` surface reporting effective posture; negative
   tests pin that no-policy + unavailable-backend refuses execution.
3. **FOU-S2**: container sandbox promotion into core behind the capability loader, with
   graceful degradation to the OS sandbox.

Rollout work is tracked in the FOU ledger per the ADR README rule (ADRs record decisions;
the ledger records work). The default flips only after S1a's qualification suite is green on
the CI matrix (Linux + macOS), mirroring the native-parity promotion pattern.

## Alternatives Considered

- **Keep `preferred` as default and only harden backends**: rejected; a fail-open default is
  the gap this ADR exists to close.
- **Make `preferred` fail-closed (rename semantics)**: rejected; `preferred`'s soft semantics
  are documented and used, and overloading it would break existing explicit users.
- **Container-only sandboxing**: rejected as the sole mechanism; Docker is not universally
  available and the OS namespace sandbox is the right minimal fallback.

## References

- `victor/tools/sandbox/backends.py` — `effective_sandbox_policy()`, `resolve_sandbox_backend()`
- `victor/config/sandbox_settings.py` — policy and knob definitions
- `victor/config/security_settings.py` — `code_executor_*` container defaults
- `verticals/victor-coding/victor_coding/tools/code_executor_tool.py` — `CodeSandbox` quality bar
- `docs/architecture/victor-foundations-program.md` — FOU Tier-S ordering

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-11-16 | 1.0 | Initial ADR | Vijaykumar Singh |
