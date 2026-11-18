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

"""Quickstart orchestration: detect, choose, and provision a local inference server.

Victor is a thin agent runtime — it calls deployed servers and never embeds
inference. ``victor init`` runs this quickstart as its first step so a new
user reaches a working ``victor chat "hello"`` with zero manual setup:

1. **Detect running servers** (short-timeout HTTP probes):
   Ollama (``/api/tags``), llama.cpp ``llama-server`` (``/health``), and
   InferFlux (``/healthz``) — llama.cpp and InferFlux share port 8080 and are
   distinguished by endpoint/response shape.
2. **Ollama detected** → verify the default model is installed; offer to pull
   it (auto-yes in non-interactive mode) via ``OllamaProvider.pull_model``.
3. **Nothing detected** → LAST RESORT: download the official llama.cpp
   ``llama-server`` release binary plus a small GGUF model into
   ``~/.victor/models/``, start ``llama-server`` as a detached daemon
   (``subprocess.Popen(start_new_session=True)`` with a PID file), and poll
   ``/health`` until ready. The daemon path is idempotent: an already-running
   managed daemon (PID file + healthy ``/health``) is reused, never restarted.
4. **Write the active profile** through :class:`~victor.config.profiles.ProfileManager`.
5. **Smoke test**: one chat turn asking the model to reply ``SOAK-OK``, sent
   through the server's OpenAI-compatible ``/v1/chat/completions`` endpoint
   (the same transport the chat loop uses for local servers).

All network I/O is httpx with explicit timeouts; every spawn records a PID
file; nothing here hits the network without the caller opting in.
"""

from __future__ import annotations

import asyncio
import platform
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from rich.console import Console
from rich.prompt import Confirm

from victor.config.profiles import ProfileManager
from victor.ui.console import get_console

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Running Ollama server (default port).
OLLAMA_BASE_URL = "http://localhost:11434"

#: llama.cpp ``llama-server`` default port (also used by InferFlux tunnels).
LLAMACPP_BASE_URL = "http://localhost:8080"

#: Model quickstart ensures for the Ollama path (matches the bundled default).
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"

#: Small instruct GGUF (~4.4 GB Q4_K_M) from bartowski's canonical quantized
#: mirror of Qwen/Qwen2.5-Coder-7B-Instruct.
LLAMACPP_GGUF_FILENAME = "qwen2.5-coder-7b-instruct-Q4_K_M.gguf"
LLAMACPP_GGUF_URL = (
    "https://huggingface.co/bartowski/qwen2.5-coder-7b-instruct-GGUF/resolve/main/"
    + LLAMACPP_GGUF_FILENAME
)

#: llama.cpp release channel. Release asset names embed the build tag
#: (e.g. ``llama-b5678-bin-ubuntu-x64.zip``), so the exact binary URL cannot
#: be a static constant — it is resolved at runtime from the latest release's
#: asset list (see :func:`select_release_asset` and
#: :data:`LLAMACPP_ASSET_PATTERNS`).
LLAMACPP_RELEASES_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

#: Per-platform asset-name substrings, in preference order. Keys use the
#: normalized ``(system, machine)`` values produced by :func:`_platform_key`.
LLAMACPP_ASSET_PATTERNS: Dict[Tuple[str, str], List[str]] = {
    ("linux", "x86_64"): ["bin-ubuntu-x64"],
    ("linux", "aarch64"): ["bin-ubuntu-arm64"],
    ("darwin", "aarch64"): ["bin-macos-arm64"],
    ("darwin", "x86_64"): ["bin-macos-x64"],
    # CUDA build preferred; CPU build is the fallback.
    ("windows", "x86_64"): ["bin-win-cuda", "bin-win-cpu"],
}

#: Timeout for reachability probes (matches doctor's 2s budget).
SERVER_PROBE_TIMEOUT = 2.0

#: Bounded wait for a freshly started llama-server to answer /health.
DAEMON_READY_TIMEOUT_SECONDS = 120.0

#: Poll cadence while waiting for daemon readiness.
DAEMON_POLL_INTERVAL_SECONDS = 1.0

#: HTTP timeout for model/binary downloads.
DOWNLOAD_TIMEOUT_SECONDS = 600.0

#: Filenames written into ~/.victor/.
LLAMACPP_PID_FILENAME = "llama-server.pid"
LLAMACPP_LOG_FILENAME = "llama-server.log"
MODEL_DIRNAME = "models"

#: Profile written for the quickstart-provisioned llama-server.
LLAMACPP_PROFILE_NAME = "local-llamacpp"

#: InferFlux profile already ships in the bundled defaults.
INFERFLUX_PROFILE_NAME = "inferflux"
INFERFLUX_DEFAULT_MODEL = "qwen3-coder-30b"

#: One-turn smoke prompt and expected marker.
SMOKE_PROMPT = "Reply with exactly: SOAK-OK"
SMOKE_MARKER = "SOAK-OK"

#: llama.cpp binary name per platform.
_LLAMACPP_BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DetectedServer:
    """A local inference server discovered by a reachability probe."""

    name: str
    kind: str  # "ollama" | "llamacpp" | "inferflux"
    base_url: str
    models: List[str] = field(default_factory=list)


@dataclass
class QuickstartResult:
    """Outcome of a quickstart run.

    ``status`` is one of ``"ready"`` (a server is configured and healthy),
    ``"skipped"`` (the user declined provisioning), or ``"failed"``.
    """

    status: str
    provider: Optional[str] = None
    model: Optional[str] = None
    smoke_ok: bool = False
    detail: str = ""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _model_matches(installed: List[str], wanted: str) -> bool:
    """True if ``wanted`` is installed, honoring Ollama's ``:latest`` default.

    A tag-less request (``qwen2.5-coder``) is satisfied by the ``:latest``
    install; a tagged request (``qwen2.5-coder:7b``) requires the exact tag —
    ``qwen2.5-coder:latest`` must NOT satisfy it.
    """
    wanted = wanted.strip()
    for name in installed:
        name = name.strip()
        if name == wanted:
            return True
        if ":" in wanted:
            continue
        base, _, tag = name.partition(":")
        if base == wanted and (tag == "latest" or not tag):
            return True
    return False


async def probe_ollama(
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = SERVER_PROBE_TIMEOUT,
) -> Optional[DetectedServer]:
    """Probe an Ollama server via ``/api/tags`` and parse installed models."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(base_url + "/api/tags")
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001 — any probe failure means "not running"
        return None
    models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
    return DetectedServer(name="ollama", kind="ollama", base_url=base_url, models=models)


async def probe_llamacpp(
    base_url: str = LLAMACPP_BASE_URL,
    timeout: float = SERVER_PROBE_TIMEOUT,
) -> Optional[DetectedServer]:
    """Probe a llama.cpp ``llama-server`` via ``/health``.

    llama.cpp answers ``{"status": "ok"}``; that exact shape is what
    distinguishes it from InferFlux on the same port.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(base_url + "/health")
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if payload.get("status") != "ok":
        return None
    return DetectedServer(name="llama.cpp server", kind="llamacpp", base_url=base_url)


async def probe_inferflux(
    base_url: str = LLAMACPP_BASE_URL,
    timeout: float = SERVER_PROBE_TIMEOUT,
) -> Optional[DetectedServer]:
    """Probe an InferFlux server via ``/healthz``.

    InferFlux serves ``/healthz`` (llama.cpp serves ``/health``), so a 200
    here with no llama.cpp ``{"status": "ok"}`` shape on ``/health`` ids the
    InferFlux tunnel.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(base_url + "/healthz")
        if response.status_code != 200:
            return None
    except Exception:  # noqa: BLE001
        return None
    return DetectedServer(name="InferFlux", kind="inferflux", base_url=base_url)


async def detect_servers() -> List[DetectedServer]:
    """Probe all known local server endpoints concurrently.

    Returns only the servers that answered, ordered by quickstart preference:
    Ollama first, then a foreign llama.cpp server, then InferFlux.
    """
    results = await asyncio.gather(
        probe_ollama(),
        probe_llamacpp(),
        probe_inferflux(),
        return_exceptions=True,
    )
    found = [r for r in results if isinstance(r, DetectedServer)]
    priority = {"ollama": 0, "llamacpp": 1, "inferflux": 2}
    found.sort(key=lambda s: priority.get(s.kind, 99))
    return found


# ---------------------------------------------------------------------------
# Ollama model provisioning
# ---------------------------------------------------------------------------


async def ensure_ollama_model(
    model: str = DEFAULT_OLLAMA_MODEL,
    *,
    assume_yes: bool = False,
    console: Optional[Console] = None,
    base_url: str = OLLAMA_BASE_URL,
) -> bool:
    """Ensure ``model`` is installed on the Ollama server; pull it if missing.

    Interactive mode asks before pulling (the download is gigabytes);
    ``assume_yes`` (non-interactive) pulls without asking. Reuses
    :meth:`OllamaProvider.pull_model` so quickstart tracks the same pull
    progress protocol as the provider layer.

    Returns:
        True when the model is available after the call.
    """
    console = console or get_console()
    try:
        async with httpx.AsyncClient(timeout=SERVER_PROBE_TIMEOUT) as client:
            response = await client.get(base_url + "/api/tags")
        installed = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]![/] Could not list Ollama models: {exc}")
        return False

    if _model_matches(installed, model):
        return True

    if not assume_yes:
        if not Confirm.ask(
            f"[cyan]Model '{model}' is not installed. Pull it now " f"(multi-GB download)?[/]",
            default=True,
        ):
            return False

    # Reuse the provider's pull API — same endpoint, same progress protocol.
    from victor.providers.ollama_provider import OllamaProvider
    from victor.providers.base import ProviderError

    provider = OllamaProvider(base_url=base_url, timeout=600)
    try:
        console.print(f"[dim]Pulling {model}…[/]")
        async for progress in provider.pull_model(model):
            status = progress.get("status")
            if status:
                console.print(f"[dim]  {status}[/]")
        return True
    except ProviderError as exc:
        console.print(f"[yellow]![/] Model pull failed: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]![/] Model pull failed: {exc}")
        return False
    finally:
        try:
            await provider.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# llama.cpp daemon management
# ---------------------------------------------------------------------------


def daemon_pid_file(victor_dir: Path) -> Path:
    """PID file for the quickstart-managed llama-server daemon."""
    return victor_dir / LLAMACPP_PID_FILENAME


def daemon_log_file(victor_dir: Path) -> Path:
    """Log file for the quickstart-managed llama-server daemon."""
    return victor_dir / LLAMACPP_LOG_FILENAME


def model_dir(victor_dir: Path) -> Path:
    """Directory where quickstart downloads binaries and GGUF models."""
    return victor_dir / MODEL_DIRNAME


async def is_managed_daemon_running(victor_dir: Path) -> bool:
    """True when the quickstart daemon is already up (PID file + healthy).

    The PID file alone is not evidence of liveness — a stale PID could have
    been recycled or the process could have died — so ``/health`` must also
    answer before the daemon path is skipped as idempotent.
    """
    pid_file = daemon_pid_file(victor_dir)
    if not pid_file.exists():
        return False
    try:
        async with httpx.AsyncClient(timeout=SERVER_PROBE_TIMEOUT) as client:
            response = await client.get(LLAMACPP_BASE_URL + "/health")
        if response.status_code != 200:
            return False
        return response.json().get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def start_llamacpp_daemon(
    binary: Path,
    model_path: Path,
    *,
    pid_file: Path,
    log_file: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> int:
    """Start ``llama-server`` as a detached daemon and record its PID.

    The process is detached via ``start_new_session=True`` so it survives the
    CLI exiting and is not tied to this process's process group. Returns the
    daemon PID; the caller polls ``/health`` for readiness.
    """
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "ab")
    try:
        process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [
                str(binary),
                "-m",
                str(model_path),
                "--host",
                host,
                "--port",
                str(port),
                "--jinja",
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    pid_file.write_text(str(process.pid))
    return process.pid


async def wait_for_llamacpp_health(
    base_url: str = LLAMACPP_BASE_URL,
    timeout_seconds: float = DAEMON_READY_TIMEOUT_SECONDS,
    poll_interval: float = DAEMON_POLL_INTERVAL_SECONDS,
) -> bool:
    """Poll ``/health`` until the server answers, bounded by ``timeout_seconds``."""
    deadline = time.monotonic() + timeout_seconds
    async with httpx.AsyncClient(timeout=SERVER_PROBE_TIMEOUT) as client:
        while True:
            try:
                response = await client.get(base_url + "/health")
                if response.status_code == 200 and response.json().get("status") == "ok":
                    return True
            except Exception:  # noqa: BLE001 — not ready yet
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(poll_interval, remaining))


# ---------------------------------------------------------------------------
# llama.cpp provisioning (binary + GGUF download)
# ---------------------------------------------------------------------------


def _platform_key() -> Tuple[str, str]:
    """(system, machine) normalized for :data:`LLAMACPP_ASSET_PATTERNS`."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "aarch64"
    if system == "windows":
        system = "windows"
    return (system, machine)


def select_release_asset(
    release_json: Dict[str, Any],
    *,
    system: str,
    machine: str,
) -> Optional[str]:
    """Pick the llama.cpp release asset download URL for this platform.

    ``system``/``machine`` are normalized the same way as
    :func:`_platform_key`, so raw ``platform.system()``/``platform.machine()``
    values (e.g. ``"AMD64"``) are accepted.

    Returns the ``browser_download_url`` of the first asset matching the
    platform's preference list, or None when nothing matches.
    """
    machine = machine.lower()
    if machine in ("amd64", "x64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "aarch64"
    patterns = LLAMACPP_ASSET_PATTERNS.get((system.lower(), machine), [])
    for pattern in patterns:
        for asset in release_json.get("assets", []):
            name = asset.get("name", "")
            if pattern in name and (name.endswith(".zip") or name.endswith(".tar.gz")):
                return asset.get("browser_download_url")
    return None


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    console: Optional[Console] = None,
) -> Path:
    """Stream ``url`` to ``dest`` (parent directories created)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with open(dest, "wb") as fh:
            async for chunk in response.aiter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
    if console is not None:
        size_mb = dest.stat().st_size / (1 << 20)
        console.print(f"[dim]  Downloaded {dest.name} ({size_mb:.1f} MB)[/]")
    return dest


def _safe_extract_zip(archive: Path, dest_dir: Path) -> None:
    """Extract a zip archive, rejecting members that escape ``dest_dir``."""
    with zipfile.ZipFile(archive) as zf:
        dest_resolved = dest_dir.resolve()
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"archive member escapes destination: {member.filename}")
        zf.extractall(dest_dir)


def _safe_extract_tar(archive: Path, dest_dir: Path) -> None:
    """Extract a tar.gz archive, rejecting members that escape ``dest_dir``."""
    with tarfile.open(archive, "r:gz") as tf:
        dest_resolved = dest_dir.resolve()
        for member in tf.getmembers():
            target = (dest_dir / member.name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"archive member escapes destination: {member.name}")
        tf.extractall(dest_dir, filter="data")


def _find_llamacpp_binary(root: Path) -> Optional[Path]:
    """Locate the ``llama-server`` binary under ``root``."""
    matches = sorted(p for p in root.rglob(_LLAMACPP_BINARY_NAME) if p.is_file())
    if not matches:
        return None
    binary = matches[0]
    if sys.platform != "win32":
        binary.chmod(0o755)
    return binary


async def ensure_llamacpp_provisioned(
    victor_dir: Path,
    console: Console,
) -> Tuple[Path, Path]:
    """Ensure the llama-server binary and GGUF model exist under ``victor_dir``.

    Downloads (idempotent — existing files are reused):
      - GGUF from :data:`LLAMACPP_GGUF_URL` into ``~/.victor/models/``
      - the latest official llama.cpp release archive (resolved via
        :data:`LLAMACPP_RELEASES_API_URL`) into ``~/.victor/models/llamacpp/``

    Returns:
        ``(llama_server_binary, gguf_path)``.

    Raises:
        RuntimeError: when no release asset matches this platform or the
            archive does not contain a ``llama-server`` binary.
    """
    models = model_dir(victor_dir)
    gguf_path = models / LLAMACPP_GGUF_FILENAME

    tool_root = models / "llamacpp"
    tool_root.mkdir(parents=True, exist_ok=True)
    binary = _find_llamacpp_binary(tool_root)

    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
        if not gguf_path.exists():
            console.print(f"[dim]Downloading model ({LLAMACPP_GGUF_FILENAME})…[/]")
            await download_file(client, LLAMACPP_GGUF_URL, gguf_path, console)
        else:
            console.print(f"[dim]Model already present: {gguf_path}[/]")

        if binary is None:
            console.print("[dim]Resolving latest llama.cpp release…[/]")
            release_response = await client.get(LLAMACPP_RELEASES_API_URL)
            release_response.raise_for_status()
            system, machine = _platform_key()
            asset_url = select_release_asset(
                release_response.json(), system=system, machine=machine
            )
            if asset_url is None:
                raise RuntimeError(
                    f"No llama.cpp release asset matches platform {system}/{machine}; "
                    "install llama-server manually from https://github.com/ggml-org/llama.cpp"
                )
            archive_name = asset_url.rsplit("/", 1)[-1]
            archive_path = tool_root / archive_name
            console.print(f"[dim]Downloading {archive_name}…[/]")
            await download_file(client, asset_url, archive_path, console)
            if archive_name.endswith(".zip"):
                _safe_extract_zip(archive_path, tool_root)
            else:
                _safe_extract_tar(archive_path, tool_root)
            archive_path.unlink(missing_ok=True)
            binary = _find_llamacpp_binary(tool_root)
            if binary is None:
                raise RuntimeError(
                    f"llama.cpp release archive did not contain {_LLAMACPP_BINARY_NAME}"
                )
        else:
            console.print(f"[dim]llama-server already present: {binary}[/]")

    return binary, gguf_path


# ---------------------------------------------------------------------------
# Profile writing
# ---------------------------------------------------------------------------


def write_profile(
    config_dir: Path,
    *,
    name: str,
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    description: Optional[str] = None,
    set_default: bool = True,
) -> Path:
    """Write a profile through :class:`ProfileManager` and make it the default.

    Preserves all other profiles already present in ``profiles.yaml``. When
    ``base_url`` is given it is stored under ``providers.<provider>.base_url``
    (the same shape the bundled defaults document for llama.cpp).
    """
    manager = ProfileManager(config_dir=config_dir)
    data = manager.load_profiles()

    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[name] = {
        "provider": provider,
        "model": model,
        "temperature": 0.6,
        "max_tokens": 4096,
        "description": description or f"{provider} {model} (set up by victor init quickstart)",
    }
    data["profiles"] = profiles

    if base_url:
        providers = data.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        provider_cfg = providers.get(provider)
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}
        provider_cfg["base_url"] = base_url
        providers[provider] = provider_cfg
        data["providers"] = providers

    if set_default:
        data["default_profile"] = name

    manager.save_profiles(data)
    return manager.profiles_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


async def run_smoke_chat(
    base_url: str,
    model: str,
    *,
    timeout: float = 60.0,
) -> Tuple[bool, str]:
    """One chat turn through the server's OpenAI-compatible endpoint.

    Sends :data:`SMOKE_PROMPT` and checks for :data:`SMOKE_MARKER` in the
    reply. This is the same ``/v1/chat/completions`` transport the chat loop
    uses for local servers (Ollama and llama-server both expose it), kept
    one-turn so quickstart stays fast and deterministic.

    Returns:
        ``(ok, reply_or_error)``
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code} from {url}"
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return False, f"smoke request failed: {exc}"
    return (SMOKE_MARKER in content), content.strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_quickstart(
    *,
    assume_yes: bool = False,
    console: Optional[Console] = None,
    victor_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
    run_smoke: bool = True,
) -> QuickstartResult:
    """Run the full detect → provision → configure → smoke sequence.

    Args:
        assume_yes: Non-interactive mode — accept pulls/downloads/smoke
            without prompting.
        console: Rich console for progress output.
        victor_dir: Victor config directory for PID/log/model files
            (defaults to ``~/.victor``).
        config_dir: Directory holding ``profiles.yaml`` (defaults to
            ``victor_dir``).
        run_smoke: Whether to offer/run the one-turn smoke chat.

    Returns:
        :class:`QuickstartResult` describing what was set up.
    """
    from victor.config.settings import get_project_paths

    console = console or get_console()
    if victor_dir is None:
        victor_dir = get_project_paths().global_victor_dir
    if config_dir is None:
        config_dir = victor_dir

    console.print("[bold]Quickstart: local inference server[/]")

    servers = await detect_servers()

    # --- Ollama detected: ensure model, default profile already matches. ---
    ollama = next((s for s in servers if s.kind == "ollama"), None)
    if ollama is not None:
        console.print(
            f"[green]✓[/] Ollama detected at {ollama.base_url} "
            f"({len(ollama.models)} models installed)"
        )
        available = await ensure_ollama_model(
            DEFAULT_OLLAMA_MODEL,
            assume_yes=assume_yes,
            console=console,
            base_url=ollama.base_url,
        )
        if not available:
            return QuickstartResult(
                status="skipped",
                provider="ollama",
                model=DEFAULT_OLLAMA_MODEL,
                detail=f"model '{DEFAULT_OLLAMA_MODEL}' not pulled (declined or failed)",
            )
        write_profile(
            config_dir,
            name="default",
            provider="ollama",
            model=DEFAULT_OLLAMA_MODEL,
            description="Local Ollama (set up by victor init quickstart)",
        )
        return await _finish(
            console,
            assume_yes=assume_yes,
            run_smoke=run_smoke,
            provider="ollama",
            model=DEFAULT_OLLAMA_MODEL,
            base_url=ollama.base_url,
            config_dir=config_dir,
        )

    # --- Foreign llama.cpp server already running on :8080. ---
    llamacpp = next((s for s in servers if s.kind == "llamacpp"), None)
    if llamacpp is not None:
        console.print(f"[green]✓[/] llama.cpp server detected at {llamacpp.base_url}")
        write_profile(
            config_dir,
            name=LLAMACPP_PROFILE_NAME,
            provider="llamacpp",
            model="local-model",
            base_url=llamacpp.base_url + "/v1",
            description="llama.cpp llama-server (detected running)",
        )
        return await _finish(
            console,
            assume_yes=assume_yes,
            run_smoke=run_smoke,
            provider="llamacpp",
            model="local-model",
            base_url=llamacpp.base_url,
            config_dir=config_dir,
        )

    # --- InferFlux tunnel detected (bundled profile already points here). ---
    inferflux = next((s for s in servers if s.kind == "inferflux"), None)
    if inferflux is not None:
        console.print(f"[green]✓[/] InferFlux detected at {inferflux.base_url}")
        return await _finish(
            console,
            assume_yes=assume_yes,
            run_smoke=run_smoke,
            provider="inferflux",
            model=INFERFLUX_DEFAULT_MODEL,
            base_url=inferflux.base_url,
            config_dir=config_dir,
        )

    # --- Nothing detected: last-resort managed llama.cpp daemon. ---
    if await is_managed_daemon_running(victor_dir):
        console.print(
            f"[green]✓[/] Quickstart llama-server daemon already running "
            f"(PID {daemon_pid_file(victor_dir).read_text().strip()})"
        )
        daemon_model = Path(LLAMACPP_GGUF_FILENAME).stem
        write_profile(
            config_dir,
            name=LLAMACPP_PROFILE_NAME,
            provider="llamacpp",
            model=daemon_model,
            base_url=LLAMACPP_BASE_URL + "/v1",
            description="llama.cpp daemon (started by victor init quickstart)",
        )
        return await _finish(
            console,
            assume_yes=assume_yes,
            run_smoke=run_smoke,
            provider="llamacpp",
            model=daemon_model,
            base_url=LLAMACPP_BASE_URL,
            config_dir=config_dir,
        )

    if not assume_yes:
        if not Confirm.ask(
            "[cyan]No local inference server found. Download and start a "
            "llama.cpp server (~4.4 GB model download)?[/]",
            default=True,
        ):
            console.print("[dim]Skipped quickstart provisioning.[/]")
            return QuickstartResult(
                status="skipped",
                detail="no server detected and provisioning declined",
            )

    console.print("[yellow]No local server detected — provisioning llama.cpp…[/]")
    try:
        binary, gguf_path = await ensure_llamacpp_provisioned(victor_dir, console)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]![/] llama.cpp provisioning failed: {exc}")
        return QuickstartResult(status="failed", detail=str(exc))

    daemon_model = gguf_path.stem
    start_llamacpp_daemon(
        binary,
        gguf_path,
        pid_file=daemon_pid_file(victor_dir),
        log_file=daemon_log_file(victor_dir),
    )
    console.print("[dim]Waiting for llama-server to become ready (up to 120s)…[/]")
    healthy = await wait_for_llamacpp_health()
    if not healthy:
        console.print(
            "[red]![/] llama-server did not become healthy in time. "
            f"Check {daemon_log_file(victor_dir)}"
        )
        return QuickstartResult(
            status="failed",
            provider="llamacpp",
            model=daemon_model,
            detail="daemon did not answer /health within the ready timeout",
        )
    console.print(f"[green]✓[/] llama-server running (PID file: {daemon_pid_file(victor_dir)})")

    write_profile(
        config_dir,
        name=LLAMACPP_PROFILE_NAME,
        provider="llamacpp",
        model=daemon_model,
        base_url=LLAMACPP_BASE_URL + "/v1",
        description="llama.cpp daemon (started by victor init quickstart)",
    )
    return await _finish(
        console,
        assume_yes=assume_yes,
        run_smoke=run_smoke,
        provider="llamacpp",
        model=daemon_model,
        base_url=LLAMACPP_BASE_URL,
        config_dir=config_dir,
    )


async def _finish(
    console: Console,
    *,
    assume_yes: bool,
    run_smoke: bool,
    provider: str,
    model: str,
    base_url: str,
    config_dir: Path,
) -> QuickstartResult:
    """Shared tail of quickstart: report the active profile, run the smoke."""
    console.print(
        f"[green]✓[/] Active profile: {provider}/{model} "
        f"(profiles.yaml: {ProfileManager(config_dir=config_dir).profiles_path})"
    )

    smoke_ok = False
    smoke_detail = ""
    if run_smoke and (
        assume_yes or Confirm.ask("[cyan]Run a one-turn smoke test now?[/]", default=True)
    ):
        console.print("[dim]Smoke test: one chat turn…[/]")
        smoke_ok, smoke_detail = await run_smoke_chat(base_url, model)
        if smoke_ok:
            console.print("[green]✓[/] Smoke test passed: the model replied SOAK-OK")
        else:
            console.print(f"[yellow]![/] Smoke test failed: {smoke_detail}")

    return QuickstartResult(
        status="ready",
        provider=provider,
        model=model,
        smoke_ok=smoke_ok,
        detail=smoke_detail,
    )


def run_quickstart_sync(**kwargs: Any) -> QuickstartResult:
    """Sync wrapper around :func:`run_quickstart` for the CLI (typer) path."""
    return asyncio.run(run_quickstart(**kwargs))


__all__ = [
    "DEFAULT_OLLAMA_MODEL",
    "DetectedServer",
    "LLAMACPP_BASE_URL",
    "LLAMACPP_GGUF_FILENAME",
    "LLAMACPP_GGUF_URL",
    "LLAMACPP_RELEASES_API_URL",
    "OLLAMA_BASE_URL",
    "QuickstartResult",
    "SMOKE_MARKER",
    "SMOKE_PROMPT",
    "daemon_log_file",
    "daemon_pid_file",
    "detect_servers",
    "download_file",
    "ensure_llamacpp_provisioned",
    "ensure_ollama_model",
    "is_managed_daemon_running",
    "model_dir",
    "probe_inferflux",
    "probe_llamacpp",
    "probe_ollama",
    "run_quickstart",
    "run_quickstart_sync",
    "run_smoke_chat",
    "select_release_asset",
    "start_llamacpp_daemon",
    "wait_for_llamacpp_health",
    "write_profile",
]
