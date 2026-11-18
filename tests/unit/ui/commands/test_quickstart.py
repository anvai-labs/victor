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

"""Tests for the `victor init` quickstart orchestration.

All network traffic is mocked with respx and all process spawns with a fake
Popen — these tests never hit the network or start real servers.
"""

from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from victor.ui.commands import quickstart as qs
from victor.ui.commands.quickstart import (
    DEFAULT_OLLAMA_MODEL,
    LLAMACPP_BASE_URL,
    LLAMACPP_GGUF_FILENAME,
    LLAMACPP_RELEASES_API_URL,
    OLLAMA_BASE_URL,
    DetectedServer,
    QuickstartResult,
)

OLLAMA_TAGS_URL = OLLAMA_BASE_URL + "/api/tags"
OLLAMA_PULL_URL = OLLAMA_BASE_URL + "/api/pull"
LLAMACPP_HEALTH_URL = LLAMACPP_BASE_URL + "/health"
INFERFLUX_HEALTHZ_URL = LLAMACPP_BASE_URL + "/healthz"


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class FakeConfirm:
    """Rich Confirm.ask stand-in that always returns a fixed answer."""

    answer: bool = True
    calls: list = []

    @classmethod
    def ask(cls, *args, **kwargs):
        cls.calls.append(args)
        return cls.answer


class FakePopen:
    """subprocess.Popen stand-in that records invocations."""

    instances: list = []

    def __init__(self, cmd, *args, **kwargs):
        self.cmd = cmd
        self.args = args
        self.kwargs = kwargs
        self.pid = 4242
        type(self).instances.append(self)


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeConfirm.answer = True
    FakeConfirm.calls = []
    FakePopen.instances = []
    yield
    FakeConfirm.answer = True
    FakeConfirm.calls = []
    FakePopen.instances = []


@pytest.fixture()
def victor_dir(tmp_path: Path) -> Path:
    return tmp_path / "victor-home"


def make_zip_bytes(member_name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member_name, content)
    return buf.getvalue()


def mock_all_down():
    """Explicitly mock every probe endpoint as unreachable (no catch-all)."""
    respx.get(OLLAMA_TAGS_URL).mock(return_value=httpx.Response(404))
    respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(404))
    respx.get(INFERFLUX_HEALTHZ_URL).mock(return_value=httpx.Response(404))


def make_release_json(asset_name: str) -> dict:
    return {
        "tag_name": "b5678",
        "assets": [
            {
                "name": asset_name,
                "browser_download_url": f"https://github.com/ggml-org/llama.cpp/releases/download/b5678/{asset_name}",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Detection matrix
# ---------------------------------------------------------------------------


class TestDetection:
    @respx.mock
    async def test_ollama_up_parses_models(self):
        respx.get(OLLAMA_TAGS_URL).mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3:8b"}]},
            )
        )
        servers = await qs.detect_servers()
        ollama = [s for s in servers if s.kind == "ollama"]
        assert len(ollama) == 1
        assert ollama[0].base_url == OLLAMA_BASE_URL
        assert "qwen2.5-coder:7b" in ollama[0].models

    @respx.mock
    async def test_ollama_down_not_detected(self):
        respx.get(OLLAMA_TAGS_URL).mock(return_value=httpx.Response(500))
        servers = await qs.detect_servers()
        assert all(s.kind != "ollama" for s in servers)

    @respx.mock
    async def test_llamacpp_up_detected(self):
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(200, json={"status": "ok"}))
        servers = await qs.detect_servers()
        kinds = [s.kind for s in servers]
        assert "llamacpp" in kinds

    @respx.mock
    async def test_inferflux_distinguished_from_llamacpp(self):
        # InferFlux serves /healthz (not llama.cpp's {"status": "ok"} /health).
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(404))
        respx.get(INFERFLUX_HEALTHZ_URL).mock(
            return_value=httpx.Response(200, json={"status": "healthy", "service": "inferflux"})
        )
        servers = await qs.detect_servers()
        kinds = sorted(s.kind for s in servers)
        assert kinds == ["inferflux"]

    @respx.mock
    async def test_nothing_detected(self):
        respx.route(host="localhost").mock(return_value=httpx.Response(404))
        servers = await qs.detect_servers()
        assert servers == []

    @respx.mock
    async def test_connection_error_treated_as_down(self):
        respx.route(host="localhost").mock(side_effect=httpx.ConnectError("refused"))
        servers = await qs.detect_servers()
        assert servers == []


# ---------------------------------------------------------------------------
# Ollama model presence / pull
# ---------------------------------------------------------------------------


class TestEnsureOllamaModel:
    @respx.mock
    async def test_model_present_skips_pull(self):
        respx.get(OLLAMA_TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": DEFAULT_OLLAMA_MODEL}]})
        )
        pull_route = respx.post(OLLAMA_PULL_URL).mock(
            return_value=httpx.Response(200, content=b'{"status": "success"}\n')
        )
        assert await qs.ensure_ollama_model(DEFAULT_OLLAMA_MODEL, assume_yes=True) is True
        assert not pull_route.called

    @respx.mock
    async def test_model_missing_pulls_with_assume_yes(self):
        respx.get(OLLAMA_TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "other:1b"}]})
        )
        pull_route = respx.post(OLLAMA_PULL_URL).mock(
            return_value=httpx.Response(
                200,
                content=(
                    b'{"status": "pulling manifest"}\n'
                    b'{"completed": 10, "total": 100}\n'
                    b'{"status": "success"}\n'
                ),
            )
        )
        assert await qs.ensure_ollama_model(DEFAULT_OLLAMA_MODEL, assume_yes=True) is True
        assert pull_route.called

    @respx.mock
    async def test_model_missing_pull_declined_interactive(self, monkeypatch):
        respx.get(OLLAMA_TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))
        pull_route = respx.post(OLLAMA_PULL_URL).mock(
            return_value=httpx.Response(200, content=b'{"status": "success"}\n')
        )
        monkeypatch.setattr(qs, "Confirm", FakeConfirm)
        FakeConfirm.answer = False
        assert await qs.ensure_ollama_model(DEFAULT_OLLAMA_MODEL, assume_yes=False) is False
        assert not pull_route.called
        assert FakeConfirm.calls  # the user was asked

    @respx.mock
    async def test_model_name_latest_suffix_matches(self):
        respx.get(OLLAMA_TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:latest"}]})
        )
        assert await qs.ensure_ollama_model("qwen2.5-coder", assume_yes=True) is True


# ---------------------------------------------------------------------------
# llama.cpp daemon management
# ---------------------------------------------------------------------------


class TestDaemon:
    @respx.mock
    async def test_pid_file_plus_health_skips_start(self, victor_dir):
        victor_dir.mkdir(parents=True)
        pid_file = victor_dir / "llama-server.pid"
        pid_file.write_text("9999")
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(200, json={"status": "ok"}))
        assert await qs.is_managed_daemon_running(victor_dir) is True

    @respx.mock
    async def test_pid_file_without_health_not_running(self, victor_dir):
        victor_dir.mkdir(parents=True)
        (victor_dir / "llama-server.pid").write_text("9999")
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(404))
        assert await qs.is_managed_daemon_running(victor_dir) is False

    async def test_no_pid_file_not_running(self, victor_dir):
        assert await qs.is_managed_daemon_running(victor_dir) is False

    def test_start_daemon_writes_pid_file_and_detaches(self, victor_dir, tmp_path):
        victor_dir.mkdir(parents=True)
        binary = tmp_path / "llama-server"
        model = tmp_path / LLAMACPP_GGUF_FILENAME
        pid_file = victor_dir / "llama-server.pid"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(subprocess, "Popen", FakePopen)
            pid = qs.start_llamacpp_daemon(
                binary,
                model,
                pid_file=pid_file,
                log_file=victor_dir / "llama-server.log",
            )
        assert pid == 4242
        assert pid_file.read_text().strip() == "4242"
        assert len(FakePopen.instances) == 1
        popen = FakePopen.instances[0]
        assert popen.kwargs.get("start_new_session") is True
        assert str(binary) in popen.cmd
        assert str(model) in popen.cmd

    @respx.mock
    async def test_wait_for_health_success(self):
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(200, json={"status": "ok"}))
        assert await qs.wait_for_llamacpp_health(timeout_seconds=1.0) is True

    @respx.mock
    async def test_wait_for_health_timeout(self):
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(503))
        assert await qs.wait_for_llamacpp_health(timeout_seconds=0.2) is False


# ---------------------------------------------------------------------------
# Release asset selection
# ---------------------------------------------------------------------------


class TestReleaseAssetSelection:
    def test_linux_x64(self):
        release = make_release_json("llama-b5678-bin-ubuntu-x64.zip")
        assert qs.select_release_asset(release, system="linux", machine="x86_64") is not None

    def test_linux_arm64(self):
        release = make_release_json("llama-b5678-bin-ubuntu-arm64.zip")
        assert qs.select_release_asset(release, system="linux", machine="aarch64") is not None

    def test_macos_arm64(self):
        release = make_release_json("llama-b5678-bin-macos-arm64.zip")
        assert qs.select_release_asset(release, system="darwin", machine="arm64") is not None

    def test_windows_prefers_cuda(self):
        release = {
            "assets": [
                {"name": "llama-b5678-bin-win-cpu-x64.zip", "browser_download_url": "u-cpu"},
                {"name": "llama-b5678-bin-win-cuda-12.4-x64.zip", "browser_download_url": "u-cuda"},
            ]
        }
        url = qs.select_release_asset(release, system="windows", machine="AMD64")
        assert url == "u-cuda"

    def test_no_matching_asset_returns_none(self):
        release = make_release_json("llama-b5678-bin-macos-arm64.zip")
        assert qs.select_release_asset(release, system="linux", machine="x86_64") is None


# ---------------------------------------------------------------------------
# Profile writing
# ---------------------------------------------------------------------------


class TestProfileWriting:
    def test_write_default_ollama_profile(self, tmp_path):
        qs.write_profile(
            tmp_path,
            name="default",
            provider="ollama",
            model=DEFAULT_OLLAMA_MODEL,
        )
        import yaml

        data = yaml.safe_load((tmp_path / "profiles.yaml").read_text())
        assert data["profiles"]["default"]["provider"] == "ollama"
        assert data["profiles"]["default"]["model"] == DEFAULT_OLLAMA_MODEL
        assert data["default_profile"] == "default"

    def test_write_llamacpp_profile_sets_base_url(self, tmp_path):
        qs.write_profile(
            tmp_path,
            name="local-llamacpp",
            provider="llamacpp",
            model="qwen2.5-coder-7b-instruct",
            base_url=LLAMACPP_BASE_URL + "/v1",
        )
        import yaml

        data = yaml.safe_load((tmp_path / "profiles.yaml").read_text())
        assert data["profiles"]["local-llamacpp"]["provider"] == "llamacpp"
        assert data["providers"]["llamacpp"]["base_url"] == LLAMACPP_BASE_URL + "/v1"

    def test_write_profile_preserves_existing_profiles(self, tmp_path):
        import yaml

        (tmp_path / "profiles.yaml").write_text(
            yaml.safe_dump(
                {
                    "profiles": {"cloud": {"provider": "anthropic", "model": "claude"}},
                    "default_profile": "cloud",
                }
            )
        )
        qs.write_profile(tmp_path, name="default", provider="ollama", model=DEFAULT_OLLAMA_MODEL)
        data = yaml.safe_load((tmp_path / "profiles.yaml").read_text())
        assert data["profiles"]["cloud"]["provider"] == "anthropic"
        assert data["default_profile"] == "default"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestSmokeTest:
    @respx.mock
    async def test_smoke_success(self):
        respx.post(LLAMACPP_BASE_URL + "/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "SOAK-OK"}}]},
            )
        )
        ok, detail = await qs.run_smoke_chat(LLAMACPP_BASE_URL, "qwen2.5-coder-7b-instruct")
        assert ok is True
        assert "SOAK-OK" in detail

    @respx.mock
    async def test_smoke_wrong_reply_fails(self):
        respx.post(LLAMACPP_BASE_URL + "/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello there"}}]},
            )
        )
        ok, _ = await qs.run_smoke_chat(LLAMACPP_BASE_URL, "qwen2.5-coder-7b-instruct")
        assert ok is False

    @respx.mock
    async def test_smoke_server_error_fails(self):
        respx.post(LLAMACPP_BASE_URL + "/v1/chat/completions").mock(
            return_value=httpx.Response(500)
        )
        ok, _ = await qs.run_smoke_chat(LLAMACPP_BASE_URL, "qwen2.5-coder-7b-instruct")
        assert ok is False


# ---------------------------------------------------------------------------
# Full run_quickstart orchestration
# ---------------------------------------------------------------------------


class TestRunQuickstart:
    @respx.mock
    async def test_ollama_ready_flow(self, victor_dir):
        mock_all_down()
        respx.get(OLLAMA_TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": DEFAULT_OLLAMA_MODEL}]})
        )
        respx.post(OLLAMA_BASE_URL + "/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "SOAK-OK"}}]}
            )
        )
        result = await qs.run_quickstart(
            assume_yes=True, victor_dir=victor_dir, config_dir=victor_dir
        )
        assert result.status == "ready"
        assert result.provider == "ollama"
        assert result.model == DEFAULT_OLLAMA_MODEL
        assert result.smoke_ok is True

    @respx.mock
    async def test_nothing_detected_declined_skips(self, victor_dir):
        mock_all_down()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(qs, "Confirm", FakeConfirm)
            FakeConfirm.answer = False
            result = await qs.run_quickstart(
                assume_yes=False, victor_dir=victor_dir, config_dir=victor_dir
            )
        assert result.status == "skipped"
        assert FakePopen.instances == []

    @respx.mock
    async def test_managed_daemon_detected_skips_provisioning(self, victor_dir):
        victor_dir.mkdir(parents=True)
        (victor_dir / "llama-server.pid").write_text("9999")
        mock_all_down()
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(200, json={"status": "ok"}))
        respx.post(LLAMACPP_BASE_URL + "/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "SOAK-OK"}}]}
            )
        )
        result = await qs.run_quickstart(
            assume_yes=True, victor_dir=victor_dir, config_dir=victor_dir
        )
        assert result.status == "ready"
        assert result.provider == "llamacpp"
        assert FakePopen.instances == []

    @respx.mock
    async def test_full_provisioning_flow(self, victor_dir, monkeypatch):
        mock_all_down()
        # Min-size guards are real production constants; the mocked payloads
        # are tiny, so scale the guards down for this test.
        mp = monkeypatch
        mp.setattr(qs, "LLAMACPP_GGUF_MIN_BYTES", 10)
        mp.setattr(qs, "LLAMACPP_ARCHIVE_MIN_BYTES", 10)
        # Release lookup + binary download + GGUF download
        respx.get(LLAMACPP_RELEASES_API_URL).mock(
            return_value=httpx.Response(
                200, json=make_release_json("llama-b5678-bin-ubuntu-x64.zip")
            )
        )
        archive_url = (
            "https://github.com/ggml-org/llama.cpp/releases/download/b5678/"
            "llama-b5678-bin-ubuntu-x64.zip"
        )
        respx.get(archive_url).mock(
            return_value=httpx.Response(
                200, content=make_zip_bytes("llama-server", b"#!/bin/false\n")
            )
        )
        gguf_route = respx.get(host="huggingface.co").mock(
            return_value=httpx.Response(200, content=b"GGUF-fake-bytes")
        )

        def health_after_daemon_start(request):
            if FakePopen.instances:
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404)

        respx.get(LLAMACPP_HEALTH_URL).mock(side_effect=health_after_daemon_start)
        respx.post(LLAMACPP_BASE_URL + "/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "SOAK-OK"}}]}
            )
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(subprocess, "Popen", FakePopen)
            mp.setattr(qs, "_platform_key", lambda: ("linux", "x86_64"))
            result = await qs.run_quickstart(
                assume_yes=True, victor_dir=victor_dir, config_dir=victor_dir
            )

        assert result.status == "ready"
        assert result.provider == "llamacpp"
        assert result.smoke_ok is True
        assert len(FakePopen.instances) == 1
        assert gguf_route.called
        # GGUF landed in ~/.victor/models/, PID file written to ~/.victor/
        assert (victor_dir / "models" / LLAMACPP_GGUF_FILENAME).exists()
        assert (victor_dir / "llama-server.pid").exists()

    @respx.mock
    async def test_smoke_disabled(self, victor_dir):
        mock_all_down()
        respx.get(OLLAMA_TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": DEFAULT_OLLAMA_MODEL}]})
        )
        result = await qs.run_quickstart(
            assume_yes=True, victor_dir=victor_dir, config_dir=victor_dir, run_smoke=False
        )
        assert result.status == "ready"
        assert result.smoke_ok is False
        assert result.status == "ready"
        assert result.smoke_ok is False

    @respx.mock
    async def test_inferflux_only_ready(self, victor_dir):
        mock_all_down()
        respx.get(INFERFLUX_HEALTHZ_URL).mock(
            return_value=httpx.Response(200, json={"service": "inferflux"})
        )
        respx.post(LLAMACPP_BASE_URL + "/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "SOAK-OK"}}]}
            )
        )
        result = await qs.run_quickstart(
            assume_yes=True, victor_dir=victor_dir, config_dir=victor_dir
        )
        assert result.status == "ready"
        assert result.provider == "inferflux"

    async def test_result_dataclass_shape(self):
        result = QuickstartResult(status="ready", provider="ollama", model="m")
        assert result.smoke_ok is False
        assert result.detail == ""


# ---------------------------------------------------------------------------
# init.py wiring
# ---------------------------------------------------------------------------


class TestInitWiring:
    def test_no_quickstart_flag_skips_quickstart(self, monkeypatch, victor_dir):
        from victor.ui.commands import init as init_mod

        calls = []
        monkeypatch.setattr(
            init_mod,
            "_quickstart_runner",
            lambda **kwargs: calls.append(kwargs) or QuickstartResult(status="skipped"),
        )
        init_mod._maybe_run_quickstart(quickstart=False, assume_yes=False, interactive=True)
        assert calls == []

    def test_programmatic_truthy_option_does_not_trigger_quickstart(self, monkeypatch, victor_dir):
        # init() called directly (tests/scripts) passes Typer OptionInfo
        # objects, not parsed bools — quickstart must stay off so library
        # callers never trigger probes or downloads implicitly.
        from victor.ui.commands import init as init_mod

        calls = []
        monkeypatch.setattr(
            init_mod,
            "_quickstart_runner",
            lambda **kwargs: calls.append(kwargs) or QuickstartResult(status="ready"),
        )
        init_mod._maybe_run_quickstart(quickstart=object(), assume_yes=False, interactive=True)
        assert calls == []

    def test_quickstart_invoked_with_flags(self, monkeypatch, victor_dir):
        from victor.ui.commands import init as init_mod

        calls = []
        monkeypatch.setattr(
            init_mod,
            "_quickstart_runner",
            lambda **kwargs: calls.append(kwargs) or QuickstartResult(status="ready"),
        )
        init_mod._maybe_run_quickstart(quickstart=True, assume_yes=True, interactive=False)
        assert len(calls) == 1
        assert calls[0]["assume_yes"] is True
        assert calls[0]["interactive"] is False


# ---------------------------------------------------------------------------
# doctor: check_local_llama_server
# ---------------------------------------------------------------------------


class TestDoctorLlamaServerCheck:
    def _doctor(self):
        from victor.ui.commands.doctor import DoctorChecks

        return DoctorChecks()

    @respx.mock
    def test_healthy_server_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VICTOR_CONFIG_DIR", str(tmp_path))
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(200, json={"status": "ok"}))
        doctor = self._doctor()
        doctor.check_local_llama_server()
        assert doctor.checks[0].severity.value == "success"

    @respx.mock
    def test_stale_pid_file_warns(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VICTOR_CONFIG_DIR", str(tmp_path))
        (tmp_path / "llama-server.pid").write_text("9999")
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(404))
        doctor = self._doctor()
        doctor.check_local_llama_server()
        check = doctor.checks[0]
        assert check.severity.value == "warning"
        assert "9999" in check.message

    @respx.mock
    def test_no_server_info(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VICTOR_CONFIG_DIR", str(tmp_path))
        respx.get(LLAMACPP_HEALTH_URL).mock(return_value=httpx.Response(404))
        doctor = self._doctor()
        doctor.check_local_llama_server()
        assert doctor.checks[0].severity.value == "info"

    def test_check_present_in_run_all_checks(self):
        from victor.ui.commands.doctor import DoctorChecks

        assert hasattr(DoctorChecks, "check_local_llama_server")


# ---------------------------------------------------------------------------
# URL constants are real, non-placeholder URLs
# ---------------------------------------------------------------------------


class TestUrlConstants:
    def test_ollama_base_url(self):
        assert OLLAMA_BASE_URL == "http://localhost:11434"

    def test_gguf_url_points_at_huggingface(self):
        assert qs.LLAMACPP_GGUF_URL.startswith("https://huggingface.co/")
        assert LLAMACPP_GGUF_FILENAME in qs.LLAMACPP_GGUF_URL

    def test_release_api_url_is_llama_cpp_releases(self):
        assert LLAMACPP_RELEASES_API_URL == (
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        )

    def test_detected_server_dataclass(self):
        server = DetectedServer(name="ollama", kind="ollama", base_url=OLLAMA_BASE_URL)
        assert server.models == []


class TestDownloadTruncationGuard:
    async def test_truncated_download_raises_and_cleans_up(self, tmp_path):
        """A stream below the expected minimum raises and leaves no partial
        file (FEP follow-up: truncated GGUF must fail loudly)."""
        import httpx
        import respx

        from victor.ui.commands import quickstart as qs

        dest = tmp_path / "model.gguf"
        with respx.mock:
            respx.get("https://files.test/model.gguf").mock(
                return_value=httpx.Response(200, content=b"tiny")
            )
            async with httpx.AsyncClient() as client:
                with pytest.raises(ValueError, match="truncated"):
                    await qs.download_file(
                        client,
                        "https://files.test/model.gguf",
                        dest,
                        expected_min_bytes=100,
                    )
        assert not dest.exists()

    async def test_download_passes_when_above_minimum(self, tmp_path):
        import httpx
        import respx

        from victor.ui.commands import quickstart as qs

        dest = tmp_path / "model.gguf"
        with respx.mock:
            respx.get("https://files.test/model.gguf").mock(
                return_value=httpx.Response(200, content=b"x" * 200)
            )
            async with httpx.AsyncClient() as client:
                result = await qs.download_file(
                    client,
                    "https://files.test/model.gguf",
                    dest,
                    expected_min_bytes=100,
                )
        assert result == dest and dest.stat().st_size == 200
