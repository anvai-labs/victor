import asyncio
import shutil
from pathlib import Path

import pytest

from victor.tools.security_scanner_tool import scan


def _bandit_available():
    """Check if bandit is available."""
    return shutil.which("bandit") is not None


@pytest.mark.skipif(
    not _bandit_available(),
    reason="bandit not installed (install with: pip install bandit)",
)
def test_security_scan_iac_hook(tmp_path: Path):
    """Test IAC/config security scanning with bandit."""
    target = tmp_path / "bad.py"
    target.write_text("import os\nos.system('ls')\n")

    result = asyncio.run(scan(str(tmp_path), scan_types=["config"], iac_scan=True))

    assert "config" in result["results"]
    assert result["results"]["config"]["count"] >= 0


class TestBanditTimeout:
    """Co-design review item 15: the bandit invocation previously had NO
    timeout at all — a hang on a huge tree hung the scan tool call forever."""

    @pytest.mark.asyncio
    async def test_bandit_timeout_reported_without_hanging(self, tmp_path, monkeypatch):
        target = tmp_path / "bad.py"
        target.write_text("import os\nos.system('ls')\n")

        async def _fake_run_managed_process(*, argv, timeout):
            return (b"", b"", -1, True, False)

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scan(str(tmp_path), scan_types=["config"], iac_scan=True)

        assert "timed out" in result["results"]["config"]["error"].lower()

    @pytest.mark.asyncio
    async def test_bandit_failure_reports_real_stderr(self, tmp_path, monkeypatch):
        """Regression guard: the old CalledProcessError(rc, name, stderr)
        construction put stderr into .output, not .stderr, so the error
        message was always empty. The rewrite must surface the real text."""
        target = tmp_path / "bad.py"
        target.write_text("import os\nos.system('ls')\n")

        async def _fake_run_managed_process(*, argv, timeout):
            return (b"", b"unexpected bandit error text", 2, False, False)

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scan(str(tmp_path), scan_types=["config"], iac_scan=True)

        assert "unexpected bandit error text" in result["results"]["config"]["error"]
