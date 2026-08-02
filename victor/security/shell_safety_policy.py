# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Damage-scoped shell safety policy (FEP-0013).

Replaces the shell tool's binary ``readonly`` flag + hardcoded command allowlist
with a composable :class:`ShellSafetyPolicy`. The invariant is **damage-scoped**,
not command-scoped: a command is safe unless it (1) escapes the workspace,
(2) corrupts a protected asset, (3) does irreversible / cross-boundary damage,
or (4) exceeds the session's declared blast radius. ``pip install``, ``xargs``,
``sed -i`` and redirects are all safe inside the workspace; only escapes are
denied.

Layered model:

* **L0** catastrophic blocklist — :func:`victor.security.command_safety.is_dangerous_command`
  (``rm -rf /``, ``dd``, ``mkfs``, fork bombs, ``curl | sh``). Applied in every
  context; non-negotiable floor.
* **L1** workspace containment — best-effort extraction of write / network
  targets; deny any target that lands in a protected zone or escapes the
  workspace.
* **L2** context profile — blast-radius rules (network egress, backgrounding)
  per :class:`ShellSafetyProfile` (STRICT / BENCHMARK / AUTONOMOUS).

Phase 1 is **additive and opt-in**: with the default ``LEGACY`` profile the
shell tool's existing inline allowlist gate runs unchanged. A non-legacy policy
is installed per-session via :func:`configure_from_session` (wired from
``SessionConfig.shell_safety`` in ``Agent.create``) and consulted through the
session-scoped :func:`get_shell_safety_policy` accessor (a ``contextvars``
token, so parallel benchmark sessions stay isolated).
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol, Sequence, Tuple

from victor.security.command_safety import is_dangerous_command

logger = logging.getLogger(__name__)


# --- public types -----------------------------------------------------------


class SafetyVerdict(str, Enum):
    """Outcome of evaluating a shell command against a safety policy."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class ShellSafetyProfile(str, Enum):
    """Session blast-radius profile.

    Attributes:
        LEGACY: Back-compat — the shell tool's existing inline allowlist gate
            runs unchanged (no policy consultation).
        STRICT: Interactive sessions — workspace writes allowed; network egress
            and backgrounding require approval (ASK).
        BENCHMARK: Evaluation sessions — permissive inside the task repo
            (``pip install`` / build / test allowed); replaces the adapter's
            ``readonly=False`` mutation.
        AUTONOMOUS: Long-running loops — workspace writes allowed; network ASK.
        CUSTOM: Caller-supplied configuration.
    """

    LEGACY = "legacy"
    STRICT = "strict"
    BENCHMARK = "benchmark"
    AUTONOMOUS = "autonomous"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ShellCommandContext:
    """Per-call context handed to a :class:`ShellSafetyPolicy`.

    Session-level configuration (profile, protected paths, workspace root,
    network policy) lives on the *policy*, not the context — the context only
    carries what varies per command invocation.

    Attributes:
        command: The shell command string.
        cwd: Canonical working directory the command runs from.
        readonly_hint: The ``readonly`` value the caller passed (a *hint* the
            policy may honour or override).
        action_hint: The ``action`` intent the caller passed
            (``read``/``write``/``network``/``exec``).
        correlation_id: Optional correlation id for ``decision_log`` joins.
    """

    command: str
    cwd: str
    readonly_hint: Optional[bool] = None
    action_hint: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass(frozen=True)
class ShellSafetyDecision:
    """The verdict a policy returns for a command.

    Attributes:
        verdict: ALLOW / DENY / ASK.
        effective_readonly: Back-compat knob for the shell tool's downstream
            logic — ``True`` for pure reads (may hit the read cache), ``False``
            once the command writes or touches the network.
        reason: Human/actionable explanation (names the invariant violated on
            DENY — never just "not on the list").
        category: Stable RL feature key (e.g. ``"workspace_write"``,
            ``"protected_asset"``, ``"network"``).
        risk_score: 0.0–1.0 heuristic risk.
        invariant: Which of the four safety clauses was implicated, if any.
    """

    verdict: SafetyVerdict
    effective_readonly: bool = True
    reason: str = ""
    category: str = "read"
    risk_score: float = 0.0
    invariant: Optional[str] = None


# --- policy protocol --------------------------------------------------------


class ShellSafetyPolicy(Protocol):
    """Evaluate a shell command for the session's safety context.

    Interface Segregation: the shell tool depends on this protocol, not on any
    concrete implementation. Open/Closed: a new rule ships a new policy and
    composes it via :class:`CompositeShellSafetyPolicy` — no edit to the shell
    tool. Liskov: a permissive policy (BENCHMARK) substitutes for a strict one
    anywhere a ``ShellSafetyPolicy`` is expected.
    """

    name: str

    def evaluate(self, ctx: ShellCommandContext) -> ShellSafetyDecision:
        """Return the safety decision for ``ctx``."""
        ...


# --- helpers: protected zones, path classification, target extraction -------


# Filesystem zones protected in every context (invariant: protected assets).
_SYSTEM_ZONES: Tuple[str, ...] = (
    "/etc",
    "/usr",
    "/System",
    "/Library",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/sbin",
    "/bin",
)

# Commands whose path arguments are write targets (L1 containment check).
_MUTATING_BASES: frozenset = frozenset(
    {
        "rm",
        "rmdir",
        "cp",
        "mv",
        "install",
        "rsync",
        "tee",
        "ln",
        "chmod",
        "chown",
        "mkdir",
        "touch",
        "truncate",
        "unlink",
    }
)

# Commands / patterns that imply network egress (L2 blast radius).
_NETWORK_BASES: frozenset = frozenset(
    {"curl", "wget", "ssh", "scp", "sftp", "ftp", "nc", "netcat", "telnet"}
)
_NETWORK_INSTALL_RE = re.compile(
    r"\b(?:git\s+(?:clone|fetch|pull|push)|pip\d?\s+install|npm\s+install|"
    r"yarn\s+(?:add|install)|pnpm\s+(?:add|install)|uv\s+pip\s+install|"
    r"poetry\s+add)\b"
)
_URL_RE = re.compile(r"https?://", re.IGNORECASE)

# Redirect target: ``> file``, ``>> file``, ``2> file`` (skips ``>&`` dups).
_REDIRECT_RE = re.compile(r"(?:^|[\s;|(])(?:\d{1,2})?(>>|>)\s*([^\s|&;<>]+)")
# Backgrounding: trailing ``&``, ``nohup``, ``disown`` (not ``&&``).
_BG_RE = re.compile(r"(?:^|[^\s&])&\s*$|\bnohup\b|\bdisown\b")


def _victor_home() -> str:
    return os.path.normpath(os.path.expanduser("~/.victor"))


def default_protected_paths() -> Tuple[str, ...]:
    """Protected zones implicit in every session: ``~/.victor`` + system dirs."""
    zones = [_victor_home()]
    zones.extend(z for z in _SYSTEM_ZONES)
    return tuple(dict.fromkeys(os.path.normpath(z) for z in zones))


def default_aux_roots() -> Tuple[str, ...]:
    """Legitimate escape targets: the active venv, temp dirs, pip cache."""
    roots: List[str] = ["/tmp", "/var/tmp"]
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        roots.append(os.path.normpath(venv))
    cache = os.path.normpath(os.path.expanduser("~/.cache/pip"))
    roots.append(cache)
    return tuple(dict.fromkeys(os.path.normpath(r) for r in roots))


def _resolve(token: str, cwd: str) -> Optional[str]:
    """Resolve a shell target token to an absolute path, or ``None`` if not a path."""
    t = token.strip().strip("\"'").rstrip("/")
    if not t or t.startswith("&"):  # ``&1``/``&2`` fd dups are not files
        return None
    try:
        expanded = os.path.expanduser(t)
        joined = os.path.normpath(os.path.join(cwd, expanded))
    except (ValueError, OSError):
        return None
    return joined


def _classify(
    resolved: str,
    workspace_root: Optional[str],
    protected: Sequence[str],
    aux_roots: Sequence[str],
) -> str:
    """Classify a resolved path: ``protected`` / ``outside-workspace`` /
    ``aux-allowed`` / ``in-workspace``."""
    for zone in protected:
        z = zone.rstrip("/")
        if resolved == z or resolved.startswith(z + "/"):
            return "protected"
    for aux in aux_roots:
        a = aux.rstrip("/")
        if resolved == a or resolved.startswith(a + "/"):
            return "aux-allowed"
    if workspace_root:
        wr = os.path.normpath(workspace_root).rstrip("/")
        if resolved == wr or resolved.startswith(wr + "/"):
            return "in-workspace"
        return "outside-workspace"
    return "in-workspace"  # no workspace_root → assume cwd is the boundary


def _extract_write_targets(cmd: str, cwd: str) -> Tuple[List[str], bool]:
    """Best-effort extraction of write-target paths.

    Returns ``(targets, has_unrecognized_mutation)`` where ``targets`` are
    resolved absolute paths the command writes to, and ``has_unrecognized_mutation``
    flags a mutating base command whose destination could not be confidently
    classified (used for conservative profile fallbacks).
    """
    targets: List[str] = []
    # 1. Redirect targets.
    for match in _REDIRECT_RE.finditer(cmd):
        resolved = _resolve(match.group(2), cwd)
        if resolved:
            targets.append(resolved)
    # 2. Mutating-command path arguments.
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        tokens = cmd.split()
    unrecognized = False
    skip_prefixes = {"sudo", "env", "nohup", "command", "time", "timeout"}
    i = 0
    while i < len(tokens):
        base = os.path.basename(tokens[i]).lower()
        if base in skip_prefixes:
            i += 1
            continue
        if base in _MUTATING_BASES:
            collected = False
            for tok in tokens[i + 1 :]:
                if tok.startswith("-"):  # flag — skip
                    continue
                if tok.startswith("$"):  # unresolvable substitution
                    unrecognized = True
                    continue
                # Only classify path-like args (absolute, ~, or containing '/').
                if tok.startswith(("/", "~")) or "/" in tok or tok in {".", ".."}:
                    resolved = _resolve(tok, cwd)
                    if resolved:
                        targets.append(resolved)
                        collected = True
                else:
                    # bareword relative destination (e.g. ``mv a b``) → in-workspace
                    # by assumption; no protected-zone risk, so skip.
                    collected = True
            if not collected and base in {"rm", "rmdir", "mv", "cp", "chmod", "chown"}:
                unrecognized = True
        i += 1
    # De-dup preserving order.
    seen: set = set()
    uniq = [t for t in targets if not (t in seen or seen.add(t))]
    return uniq, unrecognized


def _has_network_op(cmd: str) -> bool:
    lowered = cmd.lower()
    if any(re.search(rf"\b{b}\b", lowered) for b in _NETWORK_BASES):
        return True
    if _NETWORK_INSTALL_RE.search(lowered):
        return True
    # ``git clone``/``pip install`` of a URL, or any explicit http(s) target.
    return bool(_URL_RE.search(cmd))


# --- concrete policies ------------------------------------------------------


class LegacyAllowlistPolicy:
    """Back-compat policy that defers to the shell tool's existing allowlist.

    When this is the active policy, :func:`get_shell_safety_policy` returns it
    and the shell tool runs its **existing inline** ``_validate_readonly_command``
    gate unchanged — Phase 1 default behaviour is byte-identical to pre-FEP.
    ``evaluate`` is implemented (delegating to the legacy validator via a lazy
    import to avoid a tools↔security import cycle) so a legacy layer can also
    participate in a :class:`CompositeShellSafetyPolicy` chain.
    """

    name = "legacy-allowlist"

    def evaluate(
        self, ctx: ShellCommandContext
    ) -> ShellSafetyDecision:  # pragma: no cover - thin shim
        # Lazy import: victor.tools.bash imports victor.security.* at module load.
        from victor.tools.bash import _is_dangerous, _validate_readonly_command

        if _is_dangerous(ctx.command):
            return ShellSafetyDecision(
                SafetyVerdict.DENY,
                effective_readonly=True,
                reason="Dangerous command blocked.",
                category="catastrophic",
                risk_score=1.0,
                invariant="irreversible-damage",
            )
        is_valid, failing = _validate_readonly_command(ctx.command)
        if is_valid:
            return ShellSafetyDecision(
                SafetyVerdict.ALLOW, effective_readonly=True, category="read"
            )
        return ShellSafetyDecision(
            SafetyVerdict.DENY,
            effective_readonly=True,
            reason=f"Command '{failing}' is not allowed in readonly mode.",
            category="allowlist",
        )


class DamageScopedShellSafetyPolicy:
    """L0 + L1 + L2 damage-scoped policy.

    Holds the per-session configuration (profile, workspace root, protected
    paths, network policy, auxiliary roots). ``evaluate`` applies the layered
    model to a per-call :class:`ShellCommandContext`.
    """

    name = "damage-scoped"

    def __init__(
        self,
        profile: ShellSafetyProfile = ShellSafetyProfile.STRICT,
        workspace_root: Optional[str] = None,
        protected_paths: Sequence[str] = (),
        allow_network: Optional[bool] = None,
        aux_roots: Optional[Sequence[str]] = None,
        extra_allow_patterns: Sequence[str] = (),
        deny_patterns: Sequence[str] = (),
    ) -> None:
        self._profile = ShellSafetyProfile(profile)
        self._workspace_root = os.path.normpath(workspace_root) if workspace_root else None
        self._protected: Tuple[str, ...] = tuple(
            dict.fromkeys(
                os.path.normpath(p) for p in (*default_protected_paths(), *protected_paths)
            )
        )
        self._allow_network = allow_network
        self._aux_roots: Tuple[str, ...] = tuple(
            dict.fromkeys(os.path.normpath(p) for p in (*(aux_roots or default_aux_roots()),))
        )
        self._extra_allow = tuple(re.compile(p) for p in extra_allow_patterns)
        self._deny = tuple(re.compile(p) for p in deny_patterns)

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, ctx: ShellCommandContext) -> ShellSafetyDecision:
        cmd = ctx.command or ""
        cwd = ctx.cwd or os.getcwd()

        # Caller deny patterns are absolute.
        for pat in self._deny:
            if pat.search(cmd):
                return self._deny_decision(
                    "matched session deny pattern",
                    "deny_pattern",
                    invariant=None,
                    risk=0.9,
                )

        # Caller allow patterns short-circuit to ALLOW (trusted override).
        for pat in self._extra_allow:
            if pat.search(cmd):
                return ShellSafetyDecision(
                    SafetyVerdict.ALLOW,
                    effective_readonly=False,
                    category="allow_pattern",
                    risk_score=0.1,
                )

        # L0 — catastrophic floor (every profile).
        if is_dangerous_command(cmd):
            return self._deny_decision(
                "catastrophic command blocked",
                "catastrophic",
                invariant="irreversible-damage",
                risk=1.0,
            )

        # L1 — workspace containment (write targets).
        write_targets, unrecognized_mutation = _extract_write_targets(cmd, cwd)
        for target in write_targets:
            cls = _classify(target, self._workspace_root, self._protected, self._aux_roots)
            if cls == "protected":
                return self._deny_decision(
                    f"writes to protected path '{target}'",
                    "protected_asset",
                    invariant="protected-asset",
                )
            if cls == "outside-workspace":
                return self._deny_decision(
                    f"writes outside workspace ('{target}')",
                    "workspace_escape",
                    invariant="workspace-escape",
                )
            # in-workspace / aux-allowed → permitted at L1.

        # L2 — network blast radius.
        has_network = _has_network_op(cmd)
        if has_network:
            decision = self._network_decision(cmd, write_targets)
            if decision.verdict != SafetyVerdict.ALLOW:
                return decision

        # L2 — backgrounding (STRICT only; benchmarks/autonomous loops manage it).
        if self._profile is ShellSafetyProfile.STRICT and _BG_RE.search(cmd):
            return ShellSafetyDecision(
                SafetyVerdict.ASK,
                effective_readonly=False,
                reason="backgrounding disallowed in strict mode without approval.",
                category="backgrounding",
                risk_score=0.4,
                invariant="blast-radius",
            )

        # ALLOW — conservative fallback for unrecognized mutations under STRICT.
        if unrecognized_mutation and self._profile is ShellSafetyProfile.STRICT:
            return ShellSafetyDecision(
                SafetyVerdict.ASK,
                effective_readonly=False,
                reason="mutating command with unresolvable target; approval required in strict mode.",
                category="unrecognized_mutation",
                risk_score=0.4,
            )

        effective_readonly = (not write_targets) and (not has_network)
        category = "workspace_write" if write_targets else ("network" if has_network else "read")
        return ShellSafetyDecision(
            SafetyVerdict.ALLOW,
            effective_readonly=effective_readonly,
            category=category,
            risk_score=0.1,
        )

    # -- decision builders ---------------------------------------------------

    def _deny_decision(
        self, reason: str, category: str, invariant: Optional[str], risk: float = 0.8
    ) -> ShellSafetyDecision:
        return ShellSafetyDecision(
            SafetyVerdict.DENY,
            effective_readonly=True,
            reason=reason,
            category=category,
            risk_score=risk,
            invariant=invariant,
        )

    def _network_decision(self, cmd: str, write_targets: List[str]) -> ShellSafetyDecision:
        allow = self._allow_network
        if self._profile is ShellSafetyProfile.BENCHMARK:
            # Permissive: package install / fetch inside the task repo is expected.
            return ShellSafetyDecision(
                SafetyVerdict.ALLOW,
                effective_readonly=False,
                category="network",
                risk_score=0.2,
            )
        if allow is True:
            return ShellSafetyDecision(
                SafetyVerdict.ALLOW,
                effective_readonly=False,
                category="network",
                risk_score=0.2,
            )
        # STRICT / AUTONOMOUS / CUSTOM without explicit allow → ask.
        return ShellSafetyDecision(
            SafetyVerdict.ASK,
            effective_readonly=False,
            reason="network egress requires approval in this session context.",
            category="network",
            risk_score=0.5,
            invariant="blast-radius",
        )


class CompositeShellSafetyPolicy:
    """Chain policies: short-circuit on first DENY, collapse ASK, else ALLOW.

    Mirrors the governance ``PolicyEngine`` composition semantics (FEP-0005):
    a single DENY vetoes; ASK survives only if no layer ALLOWs; ALLOW wins when
    no layer denies or asks.
    """

    name = "composite"

    def __init__(self, policies: Sequence[ShellSafetyPolicy], name: str = "composite") -> None:
        self._policies = list(policies)
        self.name = name

    def evaluate(self, ctx: ShellCommandContext) -> ShellSafetyDecision:
        allow: Optional[ShellSafetyDecision] = None
        ask: Optional[ShellSafetyDecision] = None
        for policy in self._policies:
            decision = policy.evaluate(ctx)
            if decision.verdict is SafetyVerdict.DENY:
                return decision
            if decision.verdict is SafetyVerdict.ASK and ask is None:
                ask = decision
            elif decision.verdict is SafetyVerdict.ALLOW and allow is None:
                allow = decision
        if ask is not None:
            return ask
        if allow is not None:
            return allow
        # Empty chain — default deny (fail closed).
        return ShellSafetyDecision(
            SafetyVerdict.DENY,
            reason="no policy authorized the command",
            category="empty_chain",
        )


# --- factory ----------------------------------------------------------------


def policy_for_profile(
    profile: ShellSafetyProfile | str,
    *,
    workspace_root: Optional[str] = None,
    protected_paths: Sequence[str] = (),
    allow_network: Optional[bool] = None,
    extra_allow_patterns: Sequence[str] = (),
    deny_patterns: Sequence[str] = (),
) -> ShellSafetyPolicy:
    """Build the canonical policy for a profile.

    ``LEGACY`` returns the back-compat :class:`LegacyAllowlistPolicy` (no
    workspace containment). All other profiles return a
    :class:`DamageScopedShellSafetyPolicy`.
    """
    prof = ShellSafetyProfile(profile)
    if prof is ShellSafetyProfile.LEGACY:
        return LegacyAllowlistPolicy()
    return DamageScopedShellSafetyPolicy(
        profile=prof,
        workspace_root=workspace_root,
        protected_paths=protected_paths,
        allow_network=allow_network,
        extra_allow_patterns=extra_allow_patterns,
        deny_patterns=deny_patterns,
    )


# --- session-scoped accessor (contextvar) -----------------------------------


_LEGACY_DEFAULT = LegacyAllowlistPolicy()
_SESSION_POLICY: "contextvars.ContextVar[ShellSafetyPolicy]" = contextvars.ContextVar(
    "victor_shell_safety_policy", default=_LEGACY_DEFAULT
)


def get_shell_safety_policy() -> ShellSafetyPolicy:
    """Return the active session-scoped shell safety policy.

    Defaults to :class:`LegacyAllowlistPolicy` (shell tool runs its existing
    inline allowlist gate). A non-legacy policy is installed per-session via
    :func:`set_shell_safety_policy` / :func:`configure_from_session`.
    """
    return _SESSION_POLICY.get()


def set_shell_safety_policy(
    policy: ShellSafetyPolicy,
) -> contextvars.Token[ShellSafetyPolicy]:
    """Install ``policy`` for the current session context.

    Returns the contextvar token; pass it to :func:`reset_shell_safety_policy`
    to restore the prior policy (useful in tests / nested sessions).
    """
    return _SESSION_POLICY.set(policy)


def reset_shell_safety_policy(
    token: Optional[contextvars.Token[ShellSafetyPolicy]] = None,
) -> None:
    """Reset to the prior policy (or the legacy default if ``token`` is None)."""
    if token is not None:
        _SESSION_POLICY.reset(token)
    else:
        _SESSION_POLICY.set(_LEGACY_DEFAULT)


def is_legacy_policy_active() -> bool:
    """True when the active policy is the back-compat legacy allowlist."""
    return get_shell_safety_policy().name == "legacy-allowlist"


def configure_from_session(session_config: object, settings: object = None) -> None:
    """Install the shell safety policy described by ``session_config.shell_safety``.

    Wired from ``Agent.create`` after ``session_config.apply_to_settings``. A
    ``LEGACY`` / unset profile resets to the default (no-op behaviour change).
    The workspace root defaults to the session working directory when not
    specified explicitly.
    """
    cfg = getattr(session_config, "shell_safety", None)
    if cfg is None:
        return
    profile = getattr(cfg, "profile", "legacy") or "legacy"
    if profile in ("legacy", ""):
        reset_shell_safety_policy()
        return
    workspace = getattr(cfg, "workspace_root", None)
    if not workspace:
        workspace = getattr(settings, "working_directory", None) if settings is not None else None
    protected = tuple(getattr(cfg, "protected_paths", ()) or ())
    policy = policy_for_profile(
        profile,
        workspace_root=workspace,
        protected_paths=protected,
        allow_network=getattr(cfg, "allow_network", None),
        extra_allow_patterns=tuple(getattr(cfg, "extra_allow_patterns", ()) or ()),
        deny_patterns=tuple(getattr(cfg, "deny_patterns", ()) or ()),
    )
    set_shell_safety_policy(policy)
    logger.debug("Configured shell safety policy: profile=%s workspace=%s", profile, workspace)


# re-export for typing consumers
__all__ = [
    "SafetyVerdict",
    "ShellSafetyProfile",
    "ShellCommandContext",
    "ShellSafetyDecision",
    "ShellSafetyPolicy",
    "LegacyAllowlistPolicy",
    "DamageScopedShellSafetyPolicy",
    "CompositeShellSafetyPolicy",
    "policy_for_profile",
    "get_shell_safety_policy",
    "set_shell_safety_policy",
    "reset_shell_safety_policy",
    "is_legacy_policy_active",
    "configure_from_session",
    "default_protected_paths",
    "default_aux_roots",
]
