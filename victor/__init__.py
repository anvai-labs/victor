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

"""
Victor - A universal terminal-based coding agent supporting multiple LLM providers.

Supports frontier models (Claude, GPT, Gemini) and open-source models
(Ollama, LMStudio, vLLM) with unified tool calling integration.

Package Structure (v0.4.0+):
    - victor (victor-core): Core framework, providers, orchestration
    - victor_coding: Coding vertical with Tree-sitter, LSP, code tools

Simple API (recommended):
    from victor import Agent

    agent = await Agent.create(provider="anthropic")
    result = await agent.run("Write a hello world function")
    print(result.content)

Advanced Config:
    from victor import Agent, UnifiedAgentConfig

    agent = await Agent.create(config=UnifiedAgentConfig.high_budget())

Full API (advanced):
    from victor import AgentOrchestrator, Settings

    settings = Settings()
    orchestrator = await AgentOrchestrator.from_settings(settings, "default")

For coding-specific features, install the victor-coding package:
    pip install victor-coding
Verticals are discovered automatically via the victor.plugins entry point.

Import cost:
    ``import victor`` is intentionally cheap. All public names are resolved
    lazily via PEP 562 module ``__getattr__`` on first attribute access, so
    the orchestrator/provider/framework stacks only load when actually used.
    (This supersedes the old VICTOR_LIGHT_IMPORT top-level branch; the env
    var is still honored by subpackages such as ``victor.agent``.)
"""

import importlib
import sys
from typing import Any

_MIN_SUPPORTED_PYTHON = (3, 10)


def _ensure_supported_python() -> None:
    """Fail fast with a clear error on unsupported interpreters."""
    current = sys.version_info[:3]
    if current >= _MIN_SUPPORTED_PYTHON:
        return

    required = ".".join(str(part) for part in _MIN_SUPPORTED_PYTHON)
    running = ".".join(str(part) for part in current)
    raise RuntimeError(
        f"Victor requires Python {required}+; current interpreter is Python {running}. "
        "Use Python 3.10 or newer and activate the correct virtual environment."
    )


_ensure_supported_python()

from victor._contracts_bootstrap import prefer_repo_local_victor_contracts  # noqa: E402

prefer_repo_local_victor_contracts(__file__)

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("victor-ai")
except Exception:
    __version__ = "0.0.0"  # fallback for editable installs without metadata
__author__ = "Vijaykumar Singh"
__email__ = "vijay@anvaiops.com"
__license__ = "Apache-2.0"

__all__ = [
    # Decorator API
    "agent",
    "task",
    "AgentCallable",
    "TaskDefinition",
    # Framework API (5 core concepts + supporting classes)
    "Agent",
    "Task",
    "Tools",
    "State",
    "AgentExecutionEvent",
    # Supporting classes
    "ChatSession",
    "TaskResult",
    "FrameworkTaskType",
    "ToolSet",
    "ToolCategory",
    "Stage",
    "StateHooks",
    "EventType",
    "AgentConfig",
    "UnifiedAgentConfig",
    # Errors
    "AgentError",
    "ProviderError",
    "ToolError",
    "ConfigurationError",
    "BudgetExhaustedError",
    "CancellationError",
    # Core classes (existing API)
    "AgentOrchestrator",
    "Settings",
]

# name -> (module, attribute). Resolved on first access via __getattr__ (PEP 562)
# and cached in the module globals, so `import victor` stays cheap and partial
# installs (missing optional extras) only fail when the relevant name is used.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Core classes (existing API - for backward compatibility)
    "AgentOrchestrator": ("victor.agent.orchestrator", "AgentOrchestrator"),
    "Settings": ("victor.config.settings", "Settings"),
    # Framework API (simplified - new golden path)
    "Agent": ("victor.framework", "Agent"),
    "AgentConfig": ("victor.framework", "AgentConfig"),
    "UnifiedAgentConfig": ("victor.framework", "UnifiedAgentConfig"),
    "AgentError": ("victor.framework", "AgentError"),
    "AgentExecutionEvent": ("victor.framework", "AgentExecutionEvent"),
    "BudgetExhaustedError": ("victor.framework", "BudgetExhaustedError"),
    "CancellationError": ("victor.framework", "CancellationError"),
    "ChatSession": ("victor.framework", "ChatSession"),
    "ConfigurationError": ("victor.framework", "ConfigurationError"),
    "EventType": ("victor.framework", "EventType"),
    "FrameworkTaskType": ("victor.framework", "FrameworkTaskType"),
    "ProviderError": ("victor.framework", "ProviderError"),
    "Stage": ("victor.framework", "Stage"),
    "State": ("victor.framework", "State"),
    "StateHooks": ("victor.framework", "StateHooks"),
    "Task": ("victor.framework", "Task"),
    "TaskResult": ("victor.framework", "TaskResult"),
    "ToolCategory": ("victor.framework", "ToolCategory"),
    "ToolError": ("victor.framework", "ToolError"),
    "Tools": ("victor.framework", "Tools"),
    "ToolSet": ("victor.framework", "ToolSet"),
    # Decorator API — @victor.task (@victor.agent is handled below)
    "task": ("victor.framework.decorators", "task"),
    "AgentCallable": ("victor.framework.decorators", "AgentCallable"),
    "TaskDefinition": ("victor.framework.decorators", "TaskDefinition"),
}


def __getattr__(name: str) -> Any:
    """Resolve public API names lazily (PEP 562).

    ``victor.agent`` is special: it is both a real subpackage and the
    ``@victor.agent`` decorator. The subpackage module is made callable
    (see victor/agent/__init__.py), so resolving it here to the module
    keeps both `mock.patch("victor.agent...")` and the decorator working.
    """
    if name == "agent":
        return importlib.import_module("victor.agent")
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(spec[0])
    value = getattr(module, spec[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
