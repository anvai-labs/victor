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

"""Pytest fixtures for unit tests."""

import logging
import os
import sys
from pathlib import Path

import pytest

# Pre-import victor.agent submodules that tests import dynamically (e.g. inside
# setup_method or test bodies). The tests/unit/agent/ directory mirrors the
# victor/agent/ layout and has its own __init__.py; if Python sees tests/unit/ in
# sys.path before victor/ is resolved it can bind 'agent' to the wrong package,
# causing "cannot import name X from 'agent'" errors. Importing these modules here
# at conftest load-time (before any test runs) ensures sys.modules is populated
# with the correct victor.agent.* entries and subsequent in-body imports hit the
# cache instead of doing a fresh lookup against the stale sys.path.
import victor.agent.presentation  # noqa: F401 — populates sys.modules early
import victor.agent.safety  # noqa: F401 — populates sys.modules early
import victor.agent.background_agent  # noqa: F401 — populates sys.modules early

# Ensure the real victor.framework.agent SUBMODULE is importable for patching.
import victor.framework.agent  # noqa: E402,F401 — registers the submodule in sys.modules

_UNIT_TEST_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _stable_framework_agent_submodule():
    """Pin ``victor.framework.agent`` (package attribute) to its real submodule.

    ``victor.framework`` exposes BOTH a submodule ``agent`` (which defines class
    ``Agent``) AND, via ``__getattr__``, a decorator alias ``agent`` that gets
    cached into the package ``__dict__`` on first access. Once any test triggers
    that alias, ``getattr(victor.framework, "agent")`` returns the decorator
    function for the rest of the session, so unrelated tests that
    ``patch("victor.framework.agent.Agent")`` fail with
    "'victor.framework.agent' is not a package" (observed: test_decorators,
    test_init_synthesizer fallback, test_cli — all order-dependent).

    Re-pinning the package attribute to the submodule before each test makes the
    patch target resolve deterministically. The ``@agent`` decorator is always
    imported from ``victor.framework.decorators``, so this does not affect it.
    """
    submodule = sys.modules.get("victor.framework.agent")
    if submodule is not None and getattr(submodule, "__name__", "") == "victor.framework.agent":
        import victor.framework as _framework_pkg

        _framework_pkg.__dict__["agent"] = submodule
    yield


def _safe_current_working_directory() -> Path:
    """Return a valid current working directory, recovering from deleted cwd state."""
    try:
        return Path.cwd()
    except FileNotFoundError:
        os.chdir(_UNIT_TEST_REPO_ROOT)
        return _UNIT_TEST_REPO_ROOT


@pytest.fixture(autouse=True)
def isolate_working_directory():
    """Ensure each unit test starts and ends with a valid working directory.

    Some tests chdir into temporary directories that are later deleted. When that
    leaks into the next test, imports and subprocess helpers can fail with
    FileNotFoundError during collection or runtime. Restore a safe directory after
    every test, defaulting to the repo root if the original cwd vanished.
    """
    original_cwd = _safe_current_working_directory()

    yield

    target = original_cwd if original_cwd.exists() else _UNIT_TEST_REPO_ROOT
    current_cwd = _safe_current_working_directory()
    if current_cwd != target:
        os.chdir(target)


@pytest.fixture(autouse=True)
def reset_provider_rate_limit_cache():
    """Clear shared provider rate-limit suppression state between tests.

    The provider base now shares cooldown windows by provider/model key, so this
    fixture prevents cross-test leakage while preserving in-test behavior.
    """
    from victor.providers.base import BaseProvider

    BaseProvider._rate_limit_suppression_by_key.clear()

    yield


@pytest.fixture(autouse=True)
def reset_api_keys_logger():
    """Reset the api_keys logger to ensure log propagation works for caplog.

    This fixes test isolation issues where prior tests may have configured
    the logger differently (e.g., added handlers, changed propagate flag).
    """
    # Get the logger used by api_keys module
    logger = logging.getLogger("victor.config.api_keys")

    # Store original state
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    original_propagate = logger.propagate

    # Reset to clean state for test (propagate=True ensures caplog captures)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.DEBUG)

    yield

    # Restore original state after test
    logger.handlers = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate


@pytest.fixture(autouse=True)
def reset_embedding_singleton():
    """Reset EmbeddingService singleton between tests to prevent state leakage.

    The singleton pattern means first caller wins model choice. Without
    this fixture, test ordering can cause stale embeddings from a different
    model to be returned.

    Also resets the classifier singletons (intent/task/question) which CACHE a
    reference to the EmbeddingService. Without resetting them, a classifier kept
    a reference to a service that the EmbeddingService reset had discarded — a
    source of non-deterministic embedding failures on CI when several embedding
    tests run in the same shard process.
    """
    yield
    # Only import and reset if the module was loaded (avoid importing heavy deps)
    import sys

    if "victor.storage.embeddings.service" in sys.modules:
        from victor.storage.embeddings.service import EmbeddingService

        inst = EmbeddingService._instance
        if inst is not None:
            # Reuse a REAL loaded model across tests (no weight reload). But some
            # tests patch sentence_transformers.SentenceTransformer with a MagicMock
            # whose encode.return_value is a fixed array; if _ensure_model_loaded
            # ran under that patch, self._model became that mock and leaks (every
            # later encode returns the same array -> identical embeddings). So keep
            # the model only when it is a real SentenceTransformer; otherwise drop
            # it so the next call reloads a genuine model.
            try:
                from sentence_transformers import SentenceTransformer

                is_real = isinstance(inst._model, SentenceTransformer)
            except Exception:
                is_real = False
            if is_real:
                inst._embedding_cache.clear()
                inst._current_cache_memory_bytes = 0
            else:
                EmbeddingService.reset_instance()

    # Reset classifier singletons that cache the embedding service.
    for mod_name, cls_name in (
        ("victor.storage.embeddings.intent_classifier", "IntentClassifier"),
        ("victor.storage.embeddings.task_classifier", "TaskTypeClassifier"),
        ("victor.storage.embeddings.question_classifier", "QuestionTypeClassifier"),
    ):
        module = sys.modules.get(mod_name)
        if module is None:
            continue
        cls = getattr(module, cls_name, None)
        if cls is not None and getattr(cls, "_instance", None) is not None:
            reset = getattr(cls, "reset_instance", None)
            if callable(reset):
                reset()


@pytest.fixture(autouse=True)
def reset_global_state_manager():
    """Reset GlobalStateManager singleton between tests to prevent state leakage."""
    yield
    import sys

    if "victor.state.factory" in sys.modules:
        from victor.state.factory import reset_global_manager

        reset_global_manager()


@pytest.fixture(autouse=True)
def isolate_environment_variables():
    """Isolate environment variables between tests.

    Clears API key env vars and mocks get_api_key to prevent
    tests from loading real credentials.
    """
    import os
    from unittest.mock import patch

    api_key_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "XAI_API_KEY",
        "REPLICATE_API_TOKEN",
        "HUGGINGFACE_API_KEY",
    ]
    saved = {k: os.environ.pop(k, None) for k in api_key_vars}

    with patch("victor.config.api_keys.get_api_key", return_value=None):
        yield

    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def reset_service_container():
    """Reset DI service container between tests to prevent state leakage."""
    yield
    import sys

    # Reset via container module directly (not bootstrap which may not be imported)
    if "victor.core.container" in sys.modules:
        try:
            from victor.core.container import reset_container

            reset_container()
        except ImportError:
            pass  # Module not loaded, nothing to reset
    elif "victor.core.bootstrap" in sys.modules:
        try:
            from victor.core.bootstrap import reset_container

            reset_container()
        except ImportError:
            pass


@pytest.fixture(autouse=True)
def reset_plugin_registry():
    """Reset PluginRegistry between tests to prevent plugin state leakage.

    The PluginRegistry is a singleton that caches discovered plugins.
    Without reset, the first test triggering discovery pollutes the
    registry for all subsequent tests in the same xdist worker.
    """
    yield
    import sys

    if "victor.core.plugins.registry" in sys.modules:
        try:
            from victor.core.plugins.registry import PluginRegistry

            PluginRegistry._instance = None
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_capability_registry():
    """Reset CapabilityRegistry between tests to prevent capability leakage.

    The CapabilityRegistry singleton caches registered capabilities.
    Without reset, tests that register an EditorProtocol (via vertical
    loading) pollute the registry for file_editor_tool tests.
    """
    yield
    import sys

    if "victor.core.capability_registry" in sys.modules:
        try:
            from victor.core.capability_registry import CapabilityRegistry

            instance = CapabilityRegistry.get_instance()
            if hasattr(instance, "_capabilities"):
                instance._capabilities.clear()
            if hasattr(instance, "_enhanced"):
                instance._enhanced.clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_vertical_registry():
    """Save and restore VerticalRegistry between tests.

    Full snapshot/restore prevents both pollution (new keys leaking) and
    degradation (prior tests deleting built-in entries).
    """
    import sys

    saved = None
    if "victor.core.verticals.base" in sys.modules:
        try:
            from victor.core.verticals.base import VerticalRegistry

            saved = dict(VerticalRegistry._registry)
        except (ImportError, AttributeError):
            pass
    yield
    if "victor.core.verticals.base" in sys.modules:
        try:
            from victor.core.verticals.base import VerticalRegistry

            if saved is not None:
                VerticalRegistry._registry.clear()
                VerticalRegistry._registry.update(saved)
            else:
                # Module loaded during test — remove everything it added
                VerticalRegistry._registry.clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_feature_flags():
    """Reset FeatureFlagManager singleton between tests."""
    yield
    import sys

    if "victor.core.feature_flags" in sys.modules:
        try:
            from victor.core.feature_flags import get_feature_flag_manager

            mgr = get_feature_flag_manager()
            if hasattr(mgr, "_instance"):
                type(mgr)._instance = None
        except (ImportError, AttributeError):
            pass


@pytest.fixture
def isolated_project_victor_dir(tmp_path) -> Path:
    """Provide an isolated project-local .victor directory for unit tests."""
    temp_victor = tmp_path / ".victor"
    temp_victor.mkdir(exist_ok=True)
    return temp_victor


@pytest.fixture(autouse=True)
def isolate_project_victor_storage(isolated_project_victor_dir):
    """Redirect unit-test project-local persistence into a temp .victor directory.

    Without this, code paths that rely on get_project_paths() can write to the
    real repo .victor/ directory during tests. That leaks session data into the
    shared project database and chat history files.

    Patch both project_victor_dir and project_db so history files and database
    traffic stay inside the test sandbox while other project-path behavior keeps
    using the normal cwd-based project root.
    """
    from unittest.mock import PropertyMock, patch

    from victor.config.settings import ProjectPaths

    temp_project_db = isolated_project_victor_dir / "project.db"

    with (
        patch.object(
            ProjectPaths,
            "project_victor_dir",
            new_callable=PropertyMock,
            return_value=isolated_project_victor_dir,
        ),
        patch.object(
            ProjectPaths,
            "project_db",
            new_callable=PropertyMock,
            return_value=temp_project_db,
        ),
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def isolate_global_victor_db():
    """Redirect the GLOBAL ``~/.victor`` database to a temp dir for the session.

    The global database (``~/.victor/victor.db``) holds user-wide RL data —
    ``rl_outcomes`` and the ``*_q_values`` tables that drive model/tool routing,
    including the chat banner's "Routing hint". Only the *project* database and
    API keys were isolated previously, so any test exercising the RL or session
    code paths wrote outcomes straight into the developer's real DB, skewing the
    learned routing (e.g. surfacing ``fake:fake`` / ``test-profile`` providers).

    ``DatabaseManager`` resolves ``Path.home()/.victor/victor.db`` at runtime via
    stdlib ``Path.home()`` (which honors ``$HOME``), so pointing ``HOME`` at a
    temp dir sandboxes every global-DB write without changing production code.

    Session-scoped on purpose: a per-test redirect re-created the full victor.db
    schema for every test, which both slowed the suite and produced thousands of
    throwaway temp DBs (a local disk blowout). One sandbox home per session keeps
    isolation while paying the schema cost once. Tests that need a pristine global
    DB already create their own (see the RL coordinator fixtures).

    Note: ``GLOBAL_VICTOR_DIR`` is deliberately left untouched. ``secure_paths``
    resolves the victor dir from the passwd database (anti-``$HOME``-spoofing), so
    patching the constant would diverge from ``get_victor_dir()`` and the security
    invariants. Only the database (which uses ``Path.home()``) needs redirecting.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    sandbox = tempfile.mkdtemp(prefix="victor_test_home_")
    (Path(sandbox) / ".victor").mkdir(parents=True, exist_ok=True)
    # ``global_logs_dir`` resolves via the (deliberately unpatched)
    # GLOBAL_VICTOR_DIR, not $HOME. Redirecting HOME means nothing creates the
    # real global logs dir during the session, so on a fresh CI runner tests that
    # assert it exists would fail. Ensure it exists (idempotent; matches what the
    # production bootstrap would have created).
    try:
        from victor.config.settings import GLOBAL_VICTOR_DIR

        (GLOBAL_VICTOR_DIR / "logs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    saved = {k: os.environ.get(k) for k in ("HOME", "USERPROFILE")}
    os.environ["HOME"] = sandbox
    os.environ["USERPROFILE"] = sandbox  # Windows

    try:
        from victor.core.database import reset_database

        reset_database()
    except Exception:
        pass

    yield

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        from victor.core.database import reset_database

        reset_database()
    except Exception:
        pass
    shutil.rmtree(sandbox, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_extension_cache():
    """Reset VerticalExtensionLoader caches between tests."""
    yield
    import sys

    if "victor.core.verticals.extension_loader" in sys.modules:
        try:
            from victor.core.verticals.extension_loader import VerticalExtensionLoader

            VerticalExtensionLoader._cache_manager.clear()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_change_tracker():
    """Reset FileChangeHistory singleton between tests."""
    yield
    import sys

    if "victor.agent.change_tracker" in sys.modules:
        try:
            from victor.agent.change_tracker import reset_change_tracker

            reset_change_tracker()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_mode_controller():
    """Reset AgentModeController singleton between tests."""
    yield
    import sys

    if "victor.agent.mode_controller" in sys.modules:
        try:
            from victor.agent.mode_controller import reset_mode_controller

            reset_mode_controller()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_rl_coordinator():
    """Reset RLCoordinator singleton between tests."""
    yield
    import sys

    if "victor.framework.rl.coordinator" in sys.modules:
        try:
            from victor.framework.rl.coordinator import reset_rl_coordinator

            reset_rl_coordinator()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_snapshot_store():
    """Reset FileSnapshotStore singleton between tests."""
    yield
    import sys

    if "victor.agent.snapshot_store" in sys.modules:
        try:
            from victor.agent.snapshot_store import reset_snapshot_store

            reset_snapshot_store()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_tool_cache_manager():
    """Reset ToolCacheManager singleton between tests."""
    yield
    import sys

    if "victor.tools.cache_manager" in sys.modules:
        try:
            from victor.tools.cache_manager import reset_tool_cache_manager

            reset_tool_cache_manager()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_tool_result_cache():
    """Reset ToolResultCache singleton between tests."""
    yield
    import sys

    if "victor.agent.tool_result_cache" in sys.modules:
        try:
            from victor.agent.tool_result_cache import reset_tool_cache

            reset_tool_cache()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_task_analyzer():
    """Reset TaskAnalyzer singleton between tests."""
    yield
    import sys

    if "victor.agent.task_analyzer" in sys.modules:
        try:
            from victor.agent.task_analyzer import reset_task_analyzer

            reset_task_analyzer()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_compiled_graph_cache():
    """Reset CompiledGraphCache singleton between tests."""
    yield
    import sys

    if "victor.framework.graph_cache" in sys.modules:
        try:
            from victor.framework.graph_cache import reset_compiled_graph_cache

            reset_compiled_graph_cache()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def reset_tool_metadata_registry():
    """Reset ToolMetadataRegistry singleton between tests."""
    yield
    import sys

    if "victor.tools.metadata_registry" in sys.modules:
        try:
            from victor.tools.metadata_registry import reset_global_registry

            reset_global_registry()
        except (ImportError, AttributeError):
            pass


@pytest.fixture(autouse=True)
def cleanup_dangling_asyncio_tasks():
    """Cancel leftover asyncio tasks after each test to prevent xdist hangs.

    Tests that create InMemoryEventBackend instances with connect() spawn
    background dispatch tasks. If not properly disconnected, these tasks
    keep the event loop alive, hanging xdist worker shutdown.
    """
    yield
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return  # Can't cancel tasks from inside a running loop
        if loop.is_closed():
            return
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                task.cancel()
    except RuntimeError:
        pass  # No event loop


@pytest.fixture(autouse=True)
def _reset_event_broadcaster_singleton():
    """Isolate the process-wide ``EventBroadcaster`` singleton per unit test.

    ``EventBroadcaster`` owns a loop-bound ``DeliveryEngine`` (event queue +
    broadcast/sender tasks). Across pytest's function-scoped loops a leaked
    singleton would be bound to a *closed* loop. ``reset_instance()`` disposes
    it loop-safely (drops references when the bound loop is foreign/closed).
    Shared here so EVERY bridge test file is isolated, not just one.
    """
    from victor.integrations.api.event_bridge import EventBroadcaster

    EventBroadcaster.reset_instance()
    yield
    EventBroadcaster.reset_instance()


@pytest.fixture(autouse=True, scope="session")
def set_test_mode():
    """Set TEST_MODE environment variable to redirect test telemetry.

    This fixture runs once per test session and sets TEST_MODE to prevent
    MagicMock events and test telemetry from polluting the global usage.jsonl
    file. Test events will be redirected to /tmp/victor_test_telemetry/test_usage.jsonl
    instead.

    Priority: P1 - Prevents MagicMock leakage in global logs (HIGH criticality)
    """
    import os

    original = os.environ.get("TEST_MODE")
    os.environ["TEST_MODE"] = "1"

    yield

    if original is not None:
        os.environ["TEST_MODE"] = original
    else:
        os.environ.pop("TEST_MODE", None)
