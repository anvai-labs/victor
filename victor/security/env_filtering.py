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

"""Consolidated environment variable filtering for sandboxed execution.

Single source of truth for sensitive environment variables that should
be stripped when running tools in isolation. Used by:
- victor/integrations/mcp/sandbox.py
- victor/tools/bash.py
- victor/core/plugins/external.py
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Set

# Environment variables that contain secrets or credentials.
# These are stripped from child process environments in sandbox mode.
SENSITIVE_ENV_VARS: frozenset[str] = frozenset(
    {
        # Cloud provider credentials
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        # LLM API keys
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
        "XAI_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "REPLICATE_API_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HF_TOKEN",
        # Service credentials
        "DATABASE_URL",
        "REDIS_URL",
        "MONGO_URI",
        "SLACK_TOKEN",
        "GITHUB_TOKEN",
        "GITLAB_TOKEN",
        "NPM_TOKEN",
        "PYPI_TOKEN",
        # Victor internal
        "VICTOR_SERVER_API_KEY",
        "VICTOR_SESSION_SECRET",
    }
)

# Variables safe to inherit in sandboxed contexts.
SAFE_ENV_VARS: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TERM",
        "SHELL",
        "EDITOR",
        "TMPDIR",
        "TZ",
    }
)


def get_filtered_env(
    base_env: Optional[Dict[str, str]] = None,
    extra_vars: Optional[Dict[str, str]] = None,
    strip_sensitive: bool = True,
) -> Dict[str, str]:
    """Build a filtered environment for sandboxed subprocess execution.

    Starts from ``base_env`` (or ``os.environ``), strips sensitive
    variables, and merges any ``extra_vars``.

    Args:
        base_env: Starting environment. Defaults to os.environ copy.
        extra_vars: Additional variables to inject (overrides base).
        strip_sensitive: If True (default), remove all SENSITIVE_ENV_VARS.

    Returns:
        Filtered environment dictionary safe for subprocess use.
    """
    env = dict(base_env) if base_env is not None else dict(os.environ)

    if strip_sensitive:
        for var in SENSITIVE_ENV_VARS:
            env.pop(var, None)

    if extra_vars:
        env.update(extra_vars)

    return env


def get_minimal_env(
    extra_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a minimal environment with only safe variables.

    Only inherits SAFE_ENV_VARS from the current environment,
    then merges ``extra_vars``.

    Args:
        extra_vars: Additional variables to inject.

    Returns:
        Minimal environment dictionary.
    """
    env: Dict[str, str] = {}
    for var in SAFE_ENV_VARS:
        val = os.environ.get(var)
        if val is not None:
            env[var] = val

    if extra_vars:
        env.update(extra_vars)

    return env
