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

"""Unit tests for the damage-scoped shell safety policy (FEP-0013).

The test corpus is drawn from the real blocked-command data mined from
``~/.victor/logs`` (144 rejections across rotated logs): ``xargs`` (32),
redirection ``>`` (30), ``sed -i`` (16), ``pipe to shell`` (12), ``sqlite3``
(6), ``git checkout`` (6), ``pip install`` (4). These are legitimate in-workspace
agent actions and must be ALLOWED under damage-scoping — while workspace
escapes, protected-asset writes, and catastrophic commands stay DENIED.
"""

from __future__ import annotations

import os

import pytest

from victor.security.shell_safety_policy import (
    CompositeShellSafetyPolicy,
    DamageScopedShellSafetyPolicy,
    LegacyAllowlistPolicy,
    SafetyVerdict,
    ShellCommandContext,
    ShellSafetyDecision,
    ShellSafetyProfile,
    configure_from_session,
    default_aux_roots,
    default_protected_paths,
    get_shell_safety_policy,
    is_legacy_policy_active,
    policy_for_profile,
    reset_shell_safety_policy,
    set_shell_safety_policy,
)
from victor.framework.session_config import SessionConfig, ShellSafetyConfig

REPO = "/test/repo"


@pytest.fixture(autouse=True)
def _reset_policy():
    """Ensure each test starts from the legacy default policy."""
    reset_shell_safety_policy()
    yield
    reset_shell_safety_policy()


def _ctx(cmd: str, cwd: str = REPO) -> ShellCommandContext:
    return ShellCommandContext(command=cmd, cwd=cwd)


def _strict(**kw) -> DamageScopedShellSafetyPolicy:
    return DamageScopedShellSafetyPolicy(
        profile=ShellSafetyProfile.STRICT, workspace_root=REPO, **kw
    )


def _benchmark(**kw) -> DamageScopedShellSafetyPolicy:
    return DamageScopedShellSafetyPolicy(
        profile=ShellSafetyProfile.BENCHMARK, workspace_root=REPO, **kw
    )


# --- L0: catastrophic floor (every profile) ---------------------------------


@pytest.mark.parametrize(
    "cmd",
    ["rm -rf /", "rm -rf /*", "dd if=/dev/zero of=/dev/sda", "mkfs.ext4 /dev/sda1"],
)
def test_l0_catastrophic_denied_in_every_profile(cmd):
    for profile in (
        ShellSafetyProfile.STRICT,
        ShellSafetyProfile.BENCHMARK,
        ShellSafetyProfile.AUTONOMOUS,
    ):
        pol = DamageScopedShellSafetyPolicy(profile=profile, workspace_root=REPO)
        d = pol.evaluate(_ctx(cmd))
        assert d.verdict is SafetyVerdict.DENY, f"{profile}: {cmd}"
        assert d.invariant == "irreversible-damage"


def test_fork_bomb_denied():
    d = _strict().evaluate(_ctx(":(){ :|:& };:"))
    assert d.verdict is SafetyVerdict.DENY


# --- L1: the mined corpus is ALLOWED in-workspace ---------------------------
# These are exactly the commands the legacy allowlist blocks (78/144 top blockers).


@pytest.mark.parametrize(
    "cmd",
    [
        "find . -name '*.py' | xargs grep foo",  # xargs (#1 blocker, 32)
        "echo hi > out.txt",  # redirection (#2, 30) — relative, in-workspace
        "sed -i 's/a/b/' src/mod.py",  # sed -i (#3, 16)
        "git checkout -b feature",  # git checkout (#6, 6)
        f"git checkout {REPO}/src/mod.py",
        "timeout 30 pytest -q",  # timeout (#7, 4)
        "rm build/artifact.o",  # rm in-workspace (#8, 4)
        "python3 - <<'EOF'\nprint('x')\nEOF",  # heredoc
        "ls -la | head",
        f"sqlite3 {REPO}/db.sqlite '.tables'",  # sqlite3 (#5, 6) on a repo db
    ],
)
def test_mined_corpus_allowed_in_workspace_benchmark(cmd):
    d = _benchmark().evaluate(_ctx(cmd))
    assert d.verdict is SafetyVerdict.ALLOW, f"BENCHMARK should allow in-workspace: {cmd!r} -> {d}"


def test_mined_corpus_allowed_in_workspace_strict():
    # Non-network in-workspace writes are allowed under STRICT too.
    for cmd in (
        "echo hi > out.txt",
        "sed -i 's/a/b/' src/mod.py",
        "rm build/artifact.o",
    ):
        d = _strict().evaluate(_ctx(cmd))
        assert d.verdict is SafetyVerdict.ALLOW, f"{cmd!r} -> {d}"


# --- L1: workspace escapes + protected assets denied -----------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "echo x > /etc/hosts",  # /etc is always a system-protected zone
        "rm /custom/protected/secret",  # explicit extra protected path
        "echo x > /custom/protected/cfg",
    ],
)
def test_protected_asset_writes_denied(cmd):
    pol = DamageScopedShellSafetyPolicy(
        profile=ShellSafetyProfile.BENCHMARK,
        workspace_root=REPO,
        protected_paths=("/custom/protected",),
    )
    d = pol.evaluate(_ctx(cmd))
    assert d.verdict is SafetyVerdict.DENY
    assert d.invariant == "protected-asset"
    assert d.category == "protected_asset"


def test_victor_home_directory_is_protected():
    # ``~/.victor`` (state, DBs, profiles) is protected in every session. Use
    # the same helper the policy uses so the path resolves identically regardless
    # of the test's HOME isolation.
    victor_home = default_protected_paths()[0]
    pol = DamageScopedShellSafetyPolicy(profile=ShellSafetyProfile.BENCHMARK, workspace_root=REPO)
    d = pol.evaluate(_ctx(f"rm {victor_home}/victor.db"))
    assert d.verdict is SafetyVerdict.DENY
    assert d.invariant == "protected-asset"


def test_rm_rf_under_system_dir_caught_by_l0():
    # ``rm -rf /usr/...`` matches the catastrophic ``rm -rf /`` substring, so it
    # is denied at L0 (irreversible-damage) regardless of profile — denied is
    # what matters; the layer label is catastrophic, not protected-asset.
    d = _benchmark().evaluate(_ctx("rm -rf /usr/local/bin/foo"))
    assert d.verdict is SafetyVerdict.DENY


def test_workspace_escape_denied():
    # Absolute write target outside the workspace and outside aux roots.
    d = _benchmark().evaluate(_ctx("echo x > /opt/elsewhere/out.txt"))
    assert d.verdict is SafetyVerdict.DENY
    assert d.invariant == "workspace-escape"


def test_aux_root_write_allowed():
    # Writing under /tmp (an aux root) is permitted even though it is outside
    # the workspace — legitimate tooling uses temp dirs.
    d = _benchmark().evaluate(_ctx("echo x > /tmp/victor_test_out.txt"))
    assert d.verdict is SafetyVerdict.ALLOW


def test_redirect_to_protected_zone_caught():
    d = _strict().evaluate(_ctx("cat secrets.txt > /etc/passwd"))
    assert d.verdict is SafetyVerdict.DENY
    assert d.invariant == "protected-asset"


# --- L2: network blast radius per profile -----------------------------------


def test_network_strict_asks():
    d = _strict().evaluate(_ctx("pip install requests"))
    assert d.verdict is SafetyVerdict.ASK
    assert d.category == "network"


def test_network_benchmark_allowed():
    d = _benchmark().evaluate(_ctx("pip install -e ."))
    assert d.verdict is SafetyVerdict.ALLOW
    assert d.effective_readonly is False


def test_network_explicit_allow_overrides_strict():
    pol = DamageScopedShellSafetyPolicy(
        profile=ShellSafetyProfile.STRICT, workspace_root=REPO, allow_network=True
    )
    d = pol.evaluate(_ctx("pip install requests"))
    assert d.verdict is SafetyVerdict.ALLOW


def test_curl_network_detected():
    assert _strict().evaluate(_ctx("curl https://example.com")).verdict is SafetyVerdict.ASK


def test_git_clone_is_network():
    assert _benchmark().evaluate(_ctx("git clone https://github.com/x/y")).category == "network"


# --- effective_readonly back-compat -----------------------------------------


def test_read_command_effective_readonly_true():
    d = _benchmark().evaluate(_ctx("ls -la"))
    assert d.verdict is SafetyVerdict.ALLOW
    assert d.effective_readonly is True


def test_write_command_effective_readonly_false():
    d = _benchmark().evaluate(_ctx("echo x > out.txt"))
    assert d.effective_readonly is False


# --- caller allow/deny patterns ---------------------------------------------


def test_deny_pattern_forces_deny():
    pol = DamageScopedShellSafetyPolicy(
        profile=ShellSafetyProfile.BENCHMARK,
        workspace_root=REPO,
        deny_patterns=[r"\bshutdown\b"],
    )
    d = pol.evaluate(_ctx("echo running shutdown now"))
    assert d.verdict is SafetyVerdict.DENY
    assert d.category == "deny_pattern"


def test_allow_pattern_short_circuits():
    pol = DamageScopedShellSafetyPolicy(
        profile=ShellSafetyProfile.STRICT,
        workspace_root=REPO,
        extra_allow_patterns=[r"^make\s+test"],
    )
    d = pol.evaluate(_ctx("make test"))
    assert d.verdict is SafetyVerdict.ALLOW
    assert d.category == "allow_pattern"


# --- composite policy (SOLID: open/closed + Liskov) -------------------------


def test_composite_short_circuits_on_deny():
    pol = CompositeShellSafetyPolicy([_benchmark(), _strict()])
    # Benchmark allows the write; strict denies the protected target. A DENY
    # anywhere in the chain must veto.
    d = pol.evaluate(_ctx("echo x > /etc/hosts"))
    assert d.verdict is SafetyVerdict.DENY


def test_composite_allows_when_any_layer_allows_and_none_denies():
    pol = CompositeShellSafetyPolicy([_benchmark(), _strict()])
    d = pol.evaluate(_ctx("echo x > out.txt"))
    assert d.verdict is SafetyVerdict.ALLOW


def test_composite_collapses_ask_when_no_allow():
    pol = CompositeShellSafetyPolicy([_strict()])  # strict → network ASK
    d = pol.evaluate(_ctx("pip install requests"))
    assert d.verdict is SafetyVerdict.ASK


def test_composite_empty_chain_denies():
    pol = CompositeShellSafetyPolicy([])
    assert pol.evaluate(_ctx("ls")).verdict is SafetyVerdict.DENY


def test_liskov_substitution_permissive_for_strict():
    """A permissive policy must be usable wherever a strict one is expected."""

    def run(policy: DamageScopedShellSafetyPolicy) -> SafetyVerdict:
        return policy.evaluate(_ctx("echo x > out.txt")).verdict

    assert run(_strict()) is SafetyVerdict.ALLOW
    assert run(_benchmark()) is SafetyVerdict.ALLOW


# --- session-scoped accessor (contextvar) -----------------------------------


def test_default_policy_is_legacy():
    assert is_legacy_policy_active() is True
    assert get_shell_safety_policy().name == "legacy-allowlist"


def test_set_and_reset_policy():
    token = set_shell_safety_policy(_benchmark())
    assert is_legacy_policy_active() is False
    assert get_shell_safety_policy().name == "damage-scoped"
    reset_shell_safety_policy(token)
    assert is_legacy_policy_active() is True


def test_contextvar_set_then_reset_roundtrip():
    # set installs a non-legacy policy; reset restores the legacy default.
    set_shell_safety_policy(_strict())
    assert get_shell_safety_policy().name == "damage-scoped"
    reset_shell_safety_policy()
    assert get_shell_safety_policy().name == "legacy-allowlist"


# --- factory + legacy delegation --------------------------------------------


def test_policy_for_profile_legacy_returns_legacy():
    assert isinstance(policy_for_profile(ShellSafetyProfile.LEGACY), LegacyAllowlistPolicy)


def test_policy_for_profile_strict_returns_damage_scoped():
    p = policy_for_profile("strict", workspace_root=REPO)
    assert isinstance(p, DamageScopedShellSafetyPolicy)


def test_legacy_policy_name_and_decision_shape():
    # A pure read is allowed via the legacy validator; a clear mutation is denied.
    pol = LegacyAllowlistPolicy()
    assert pol.name == "legacy-allowlist"
    allow = pol.evaluate(_ctx("ls -la"))
    assert allow.verdict is SafetyVerdict.ALLOW
    deny = pol.evaluate(_ctx("rm somefile"))
    assert deny.verdict is SafetyVerdict.DENY


# --- configure_from_session + SessionConfig integration ---------------------


def test_configure_from_session_legacy_is_noop():
    sc = SessionConfig(shell_safety=ShellSafetyConfig(profile="legacy"))
    configure_from_session(sc, None)
    assert is_legacy_policy_active() is True


def test_configure_from_session_benchmark_installs_policy():
    sc = SessionConfig(shell_safety=ShellSafetyConfig(profile="benchmark", workspace_root=REPO))
    configure_from_session(sc, None)
    assert is_legacy_policy_active() is False
    assert get_shell_safety_policy().name == "damage-scoped"


def test_configure_from_session_derives_workspace_from_settings():
    class _Settings:
        working_directory = "/derived/workspace"

    sc = SessionConfig(shell_safety=ShellSafetyConfig(profile="strict"))
    configure_from_session(sc, _Settings())
    pol = get_shell_safety_policy()
    assert isinstance(pol, DamageScopedShellSafetyPolicy)


def test_session_config_from_cli_flags_shell_safety():
    sc = SessionConfig.from_cli_flags(
        shell_safety_profile="benchmark",
        shell_workspace_root=REPO,
        shell_allow_network=True,
    )
    assert sc.shell_safety.profile == "benchmark"
    assert sc.shell_safety.workspace_root == REPO
    assert sc.shell_safety.allow_network is True


def test_session_config_from_cli_flags_invalid_profile_raises():
    with pytest.raises(ValueError):
        SessionConfig.from_cli_flags(shell_safety_profile="bogus")


def test_session_config_default_shell_safety_is_legacy():
    sc = SessionConfig()
    assert sc.shell_safety.profile == "legacy"


# --- protected/aux defaults -------------------------------------------------


def test_default_protected_paths_includes_victor_home_and_system():
    paths = default_protected_paths()
    assert any(p.endswith(".victor") for p in paths)
    assert "/etc" in paths
    assert "/usr" in paths


def test_default_aux_roots_includes_tmp():
    roots = default_aux_roots()
    assert "/tmp" in roots


# --- decision is frozen -----------------------------------------------------


def test_decision_is_frozen():
    d = ShellSafetyDecision(SafetyVerdict.ALLOW)
    with pytest.raises(Exception):
        d.verdict = SafetyVerdict.DENY  # type: ignore[misc]
