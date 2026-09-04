"""Co-design review item 15: shared subprocess runner.

Pins the two properties a naive ``process.kill()`` + ``communicate()``
never had: a timeout kills the WHOLE process group (a child the shell
command backgrounded doesn't survive as an orphan), and partial output
survives a timeout/byte-cap kill instead of being discarded.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.tools.subprocess_executor import (
    CommandErrorType,
    kill_process_group,
    run_command_async,
    run_managed_process,
)


class TestTimeoutKillsWholeProcessGroup:
    @pytest.mark.asyncio
    async def test_backgrounded_child_does_not_survive_timeout(self, tmp_path):
        """Negative test for the bug this item exists to fix: a plain
        process.kill() only kills the direct shell child, so a process the
        shell command backgrounded (``&``) is orphaned and keeps running
        past the timeout. With the process-group kill, it must not."""
        marker = tmp_path / "marker.txt"
        cmd = f"(sleep 2 && touch {marker}) & wait"

        result = await run_command_async(cmd, timeout=0.3)

        assert result.error_type == CommandErrorType.TIMEOUT
        # Wait past when the backgrounded child WOULD have finished if it
        # had merely been orphaned rather than actually killed.
        await asyncio.sleep(2.2)
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_run_managed_process_reports_timed_out(self, tmp_path):
        marker = tmp_path / "marker2.txt"
        cmd = f"(sleep 2 && touch {marker}) & wait"

        stdout, stderr, return_code, timed_out, capped = await run_managed_process(
            command=cmd, timeout=0.3
        )

        assert timed_out is True
        assert capped is False
        await asyncio.sleep(2.2)
        assert not marker.exists()


class TestPartialOutputOnTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_partial_stdout_not_empty(self):
        """Previously the async runner discarded everything and returned
        stdout="" on timeout, even though output had already been produced
        (communicate() is all-or-nothing). The incremental read loop must
        preserve whatever arrived before the kill."""
        cmd = "echo partial-before-timeout; sleep 5"

        result = await run_command_async(cmd, timeout=0.3)

        assert result.error_type == CommandErrorType.TIMEOUT
        assert "partial-before-timeout" in result.stdout


class TestMaxOutputBytesCap:
    @pytest.mark.asyncio
    async def test_cap_stops_reading_and_kills_before_completion(self):
        """A runaway process producing far more output than the cap must be
        killed once the cap is hit, not read to completion first."""
        stdout, stderr, return_code, timed_out, capped = await run_managed_process(
            command="yes | head -c 50000000",  # 50MB if allowed to finish
            timeout=10,
            max_output_bytes=100,
        )

        assert capped is True
        assert timed_out is False
        # Capped at (roughly) one read chunk past the threshold, nowhere
        # near the full 50MB the process would have produced uncapped.
        assert len(stdout) < 100_000

    @pytest.mark.asyncio
    async def test_run_command_async_marks_truncated_on_cap(self):
        result = await run_command_async("yes | head -c 50000000", timeout=10, max_output_bytes=100)

        assert result.truncated is True
        assert result.error_type != CommandErrorType.TIMEOUT
        assert len(result.stdout.encode("utf-8")) < 100_000


class TestRunManagedProcessContract:
    @pytest.mark.asyncio
    async def test_requires_exactly_one_of_command_or_argv(self):
        with pytest.raises(ValueError):
            await run_managed_process()

        with pytest.raises(ValueError):
            await run_managed_process(command="echo hi", argv=["echo", "hi"])

    @pytest.mark.asyncio
    async def test_argv_runs_without_a_shell(self):
        stdout, stderr, return_code, timed_out, capped = await run_managed_process(
            argv=["echo", "no-shell-needed"], timeout=5
        )

        assert return_code == 0
        assert stdout.decode().strip() == "no-shell-needed"
        assert timed_out is False
        assert capped is False

    @pytest.mark.asyncio
    async def test_clean_exit_returns_full_output_and_real_return_code(self):
        stdout, stderr, return_code, timed_out, capped = await run_managed_process(
            command="echo out; echo err 1>&2; exit 3", timeout=5
        )

        assert stdout.decode().strip() == "out"
        assert stderr.decode().strip() == "err"
        assert return_code == 3
        assert timed_out is False
        assert capped is False

    @pytest.mark.asyncio
    async def test_on_chunk_exception_still_kills_and_reaps_the_process(self, tmp_path):
        """A caller-supplied on_chunk that raises must not skip cleanup —
        the process (and its process group) must still be killed rather
        than left to run to completion on its own, and process.wait() in
        the finally block must not hang waiting for an unkilled child."""
        marker = tmp_path / "marker.txt"
        cmd = f"echo tick; (sleep 2 && touch {marker}) & wait"

        def _boom(is_stderr: bool, chunk: bytes) -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await run_managed_process(command=cmd, timeout=5, on_chunk=_boom)

        # Past when the backgrounded child would have finished if cleanup
        # had been skipped and it were merely left running.
        await asyncio.sleep(2.2)
        assert not marker.exists()


class TestKillProcessGroupFallback:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only killpg path")
    def test_falls_back_to_direct_kill_without_killpg(self, monkeypatch):
        """POSIX guard: when os.killpg/os.getpgid are unavailable, fall
        back to killing just the direct child rather than raising."""
        monkeypatch.delattr(os, "killpg", raising=False)
        process = MagicMock()
        process.pid = 12345
        process.returncode = None  # still running

        kill_process_group(process)

        process.kill.assert_called_once()

    def test_swallows_process_lookup_error_from_getpgid(self, monkeypatch):
        """The process may already be reaped by the time we try to kill
        its group — must not raise."""
        monkeypatch.setattr(
            os, "getpgid", MagicMock(side_effect=ProcessLookupError()), raising=False
        )
        process = MagicMock()
        process.pid = 12345
        process.returncode = None  # still running

        kill_process_group(process)  # must not raise

        process.kill.assert_called_once()

    def test_swallows_process_lookup_error_from_direct_kill(self):
        """Fallback process.kill() on an already-gone process must not
        raise either."""
        process = MagicMock()
        process.pid = 12345
        process.returncode = None  # still running
        process.kill.side_effect = ProcessLookupError()

        if hasattr(os, "killpg"):
            pytest.skip("killpg path taken on POSIX; direct-kill fallback not exercised")

        kill_process_group(process)  # must not raise

    def test_swallows_permission_error_from_direct_kill(self):
        """A PermissionError from the fallback process.kill() must not
        raise either — the primary killpg path already tolerates this."""
        process = MagicMock()
        process.pid = 12345
        process.returncode = None  # still running
        process.kill.side_effect = PermissionError()

        if hasattr(os, "killpg"):
            pytest.skip("killpg path taken on POSIX; direct-kill fallback not exercised")

        kill_process_group(process)  # must not raise


class TestKillProcessGroupReturncodeGuard:
    """PID-reuse guard: once asyncio has observed the child exit
    (process.returncode is no longer None), its pid may already have been
    recycled by the OS for an unrelated process — killpg/kill must not be
    attempted at all, mirroring the check subprocess.Popen.send_signal()
    itself applies before signaling (bpo-38630)."""

    def test_already_exited_process_is_never_signaled(self, monkeypatch):
        process = MagicMock()
        process.pid = 12345
        process.returncode = 0

        killpg_mock = MagicMock()
        monkeypatch.setattr(os, "killpg", killpg_mock, raising=False)

        kill_process_group(process)

        killpg_mock.assert_not_called()
        process.kill.assert_not_called()
