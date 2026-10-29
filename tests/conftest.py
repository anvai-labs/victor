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

"""pytest configuration and fixtures."""

import asyncio
import sys

import pytest


@pytest.fixture(autouse=True)
def reset_write_path_policy():
    """Keep global write-path policy state from leaking between tests."""
    try:
        from victor.tools.write_path_policy import set_active_write_policy

        set_active_write_policy(None)
        yield
        set_active_write_policy(None)
    except Exception:
        yield


@pytest.fixture(autouse=True)
def reset_safety_checker_singleton():
    """Keep the global safety-checker singleton from leaking between tests.

    ``get_safety_checker()`` caches a process-global ``_default_checker`` that
    freezes its approval mode at first creation and accumulates mutable state
    (confirmation callback, custom deny patterns). A test that leaves it denying
    blocks unrelated write tools in later tests — e.g. an ``edit`` execution
    returning ``success=False`` "Operation cancelled by safety check" only when
    run after the polluter (order-dependent). Reset it around every test.
    """
    try:
        from victor.agent.safety import reset_safety_checker

        reset_safety_checker()
        yield
        reset_safety_checker()
    except Exception:
        yield


@pytest.fixture(autouse=True)
def reset_tool_message_cleanup_stats():
    """Keep openai_compat tool-history-repair stats from leaking between tests.

    ``fix_orphaned_tool_messages`` records the last repair into a module ContextVar that a
    later streaming turn consumes to emit a ``tool_history_repaired`` event. Sync tests that
    call the repair directly leave that ContextVar set in the parent context, so an unrelated
    streaming test could pick up a stale repair and emit a spurious event (an order-dependent
    failure). Clear it around every test.
    """
    try:
        from victor.providers.openai_compat import clear_last_tool_message_cleanup_stats

        clear_last_tool_message_cleanup_stats()
        yield
        clear_last_tool_message_cleanup_stats()
    except Exception:
        yield


@pytest.fixture(autouse=True)
def _disable_read_before_write_gate(monkeypatch):
    """Disable the read-before-overwrite guard for the test suite by default.

    The guard (``enforce_read_before_write``, env ``VICTOR_ENFORCE_READ_BEFORE_WRITE``)
    is a *model*-safety feature that defaults ON in production: it stops an agent
    from silently clobbering an existing file it never read. In the test suite the
    test author controls every write, so that scenario does not apply and the guard
    would only break existing tests that intentionally overwrite unread files.

    Dedicated coverage lives in ``tests/unit/tools/test_read_before_write_gate.py``,
    which re-enables the guard (``monkeypatch.delenv``) per test.
    """
    monkeypatch.setenv("VICTOR_ENFORCE_READ_BEFORE_WRITE", "0")
    yield


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_victor_coding: marks tests that require victor-coding package",
    )


@pytest.fixture(autouse=True)
async def cleanup_event_bus_tasks():
    """Clean up ObservabilityBus background tasks after each test.

    Prevents pytest hangs from fire-and-forget coroutines created by
    ObservabilityBus.emit_metric(). This fixture runs automatically
    after every test to ensure proper cleanup.

    One scheduler pass lets ready callbacks run before the loop closes; the
    function-scoped ``asyncio.Runner`` then cancels and drains whatever is
    still pending. Do NOT add a per-test ``gc.collect()`` here: a full
    collection over the suite's heap cost ~1.2s per test on CI runners
    (~95% of all unit-shard wall time) and never cancelled a task.
    """
    yield
    await asyncio.sleep(0)


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests that require victor-coding if not installed."""
    # Check if victor_coding is importable
    try:
        __import__("victor_coding")
        victor_coding_available = True
    except ImportError:
        victor_coding_available = False

    # Skip tests that require victor_coding if not available
    for item in items:
        if not victor_coding_available:
            file_path = str(item.fspath)
            # file_editor tests have their own module-level skipif based on
            # runtime capability check (not just package availability)
            skip_files = [
                "test_lsp_tool",
                "test_code_intelligence_tool",
                "test_lsp.",
                "lsp_write_enhancer",
            ]
            if any(x in file_path for x in skip_files):
                item.add_marker(
                    pytest.mark.skipif(
                        not victor_coding_available,
                        reason="victor-coding package not installed",
                    )
                )


def _own_sandbox_containers() -> list:
    """Code-executor sandbox containers owned by THIS pytest process (xdist-safe)."""
    try:
        import docker

        from victor_coding.tools.code_executor_tool import (
            OWNER_HOST_LABEL,
            OWNER_PID_LABEL,
            SANDBOX_CONTAINER_LABEL,
            SANDBOX_CONTAINER_VALUE,
            host_identity,
        )

        client = docker.from_env(timeout=3)
        return client.containers.list(
            all=True,
            filters={
                "label": [
                    f"{SANDBOX_CONTAINER_LABEL}={SANDBOX_CONTAINER_VALUE}",
                    f"{OWNER_HOST_LABEL}={host_identity()}",
                    f"{OWNER_PID_LABEL}={__import__('os').getpid()}",
                ]
            },
        )
    except Exception:  # no docker / no coding vertical: nothing can have leaked
        return []


@pytest.hookimpl(optionalhook=True)  # xdist-only hook; absent without xdist
def pytest_testnodedown(node, error):
    """xdist controller: collect sandbox leaks reported by each worker."""
    leaked = getattr(node, "workeroutput", {}).get("sandbox_leaks", [])
    node.config.__dict__.setdefault("_sandbox_leaks", []).extend(leaked)


def pytest_sessionfinish(session, exitstatus):
    """Fail the session if any test left a sandbox container behind, then remove it.

    Sandbox containers are acquired lazily and released on dispose / garbage
    collection, so none should survive a session. This turns a regression of
    the 2,242-container leak into a loud failure instead of silent build-up.
    Under xdist each worker removes its own leaks and reports them to the
    controller, which fails the run.
    """
    import gc

    gc.collect()  # let finalizers of unreachable sandboxes run first
    leaked = []
    for container in _own_sandbox_containers():
        leaked.append(container.short_id)
        try:
            container.remove(force=True)
        except Exception:
            pass

    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:  # xdist worker: the controller reports and fails
        workeroutput["sandbox_leaks"] = leaked
        return

    leaked += session.config.__dict__.get("_sandbox_leaks", [])
    if not leaked:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    message = (
        f"{len(leaked)} code-executor sandbox container(s) were left running by this "
        f"test session ({', '.join(leaked)}); removed. A test acquired a CodeSandbox "
        "container and kept a reference past teardown."
    )
    if reporter is not None:
        reporter.write_line(f"SANDBOX LEAK: {message}", red=True)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
