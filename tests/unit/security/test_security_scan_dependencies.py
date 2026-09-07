import asyncio
import shutil
from pathlib import Path

import pytest

from victor.tools.security_scanner_tool import scan


def _pip_audit_available():
    """Check if pip-audit is available."""
    return shutil.which("pip-audit") is not None


@pytest.mark.skipif(
    not _pip_audit_available(),
    reason="pip-audit not installed (install with: pip install pip-audit)",
)
def test_security_scan_dependency_hook(tmp_path: Path):
    """Test dependency vulnerability scanning with pip-audit."""
    req = tmp_path / "requirements.txt"
    req.write_text("flask==0.5\n")

    result = asyncio.run(scan(str(tmp_path), scan_types=["dependencies"], dependency_scan=True))

    assert "dependencies" in result["results"]
    assert result["results"]["dependencies"]["count"] >= 0


class TestPipAuditTimeout:
    """Co-design review item 15: the pip-audit invocation previously had NO
    timeout at all — a hung network lookup or huge dependency tree hung the
    scan tool call forever."""

    @pytest.mark.asyncio
    async def test_pip_audit_timeout_reported_without_hanging(self, tmp_path, monkeypatch):
        req = tmp_path / "requirements.txt"
        req.write_text("flask==0.5\n")

        async def _fake_run_managed_process(*, argv, timeout):
            return (b"", b"", -1, True, False)

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scan(str(tmp_path), scan_types=["dependencies"], dependency_scan=True)

        assert "timed out" in result["results"]["dependencies"]["error"].lower()
        assert result["results"]["dependencies"]["count"] == 0

    @pytest.mark.asyncio
    async def test_pip_audit_failure_reports_real_stderr(self, tmp_path, monkeypatch):
        """Regression guard: the old CalledProcessError(rc, name, stderr)
        construction put stderr into .output, not .stderr, so the error
        message was always empty. The rewrite must surface the real text."""
        req = tmp_path / "requirements.txt"
        req.write_text("flask==0.5\n")

        async def _fake_run_managed_process(*, argv, timeout):
            return (b"", b"unexpected pip-audit error text", 1, False, False)

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scan(str(tmp_path), scan_types=["dependencies"], dependency_scan=True)

        assert "unexpected pip-audit error text" in result["results"]["dependencies"]["error"]
