"""Tests for victor.ui.diagnostics (faulthandler-based hang diagnostics)."""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

import pytest

from victor.ui import diagnostics


@pytest.fixture(autouse=True)
def _reset_installed_flag(monkeypatch):
    """Each test starts with diagnostics not-yet-installed."""
    monkeypatch.setattr(diagnostics, "_installed", False)
    # Default to enabled unless a test overrides it.
    monkeypatch.delenv("VICTOR_FAULTHANDLER", raising=False)
    monkeypatch.delenv("VICTOR_FAULTHANDLER_TIMEOUT", raising=False)
    yield


def test_install_returns_true_with_real_fd_stream(tmp_path):
    log = tmp_path / "fh.log"
    with log.open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is True


def test_install_is_idempotent(tmp_path):
    log = tmp_path / "fh.log"
    with log.open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is True
        # Second call is a no-op (already installed for this process).
        assert diagnostics.install_fault_diagnostics(stream=fh) is False


def test_disabled_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VICTOR_FAULTHANDLER", "0")
    log = tmp_path / "fh.log"
    with log.open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is False


@pytest.mark.parametrize("value", ["false", "No", "OFF"])
def test_disabled_accepts_truthy_variants(monkeypatch, tmp_path, value):
    monkeypatch.setenv("VICTOR_FAULTHANDLER", value)
    with (tmp_path / "fh.log").open("w") as fh:
        assert diagnostics.install_fault_diagnostics(stream=fh) is False


def test_stream_without_fileno_is_skipped_gracefully():
    # io.StringIO has no fileno(); must not raise and must report not-installed.
    assert diagnostics.install_fault_diagnostics(stream=io.StringIO()) is False


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGUSR1"), reason="SIGUSR1 unavailable")
def test_sigusr1_dumps_stack_in_subprocess():
    """Repeated stack requests must dump all threads and keep the process alive.

    Run in a subprocess so the signal handler / stack dump never disturbs the
    pytest process, and so we exercise the real faulthandler path.
    """
    script = textwrap.dedent("""
        import os, signal, sys, threading
        from victor.ui import diagnostics

        assert diagnostics.install_fault_diagnostics(stream=sys.stderr) is True
        ready = threading.Event()
        stop = threading.Event()

        def _blocked_worker_marker():
            ready.set()
            stop.wait()

        worker = threading.Thread(target=_blocked_worker_marker, daemon=True)
        worker.start()
        assert ready.wait(timeout=5)

        def _spin_marker_frame():
            # A uniquely named frame we can assert appears in the dump.
            os.kill(os.getpid(), signal.SIGUSR1)

        _spin_marker_frame()
        _spin_marker_frame()
        stop.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        print("survived both stack requests", flush=True)
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "survived both stack requests" in proc.stdout
    # Both the caller and the blocked worker must appear in each dump.
    assert proc.stderr.count("_spin_marker_frame") == 2
    assert proc.stderr.count("_blocked_worker_marker") == 2
