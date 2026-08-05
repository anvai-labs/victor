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

"""Centralized timeout configuration for Victor.

This module provides a single source of truth for all timeout values used
throughout the application. Instead of hardcoding timeout values in individual
files, reference these constants to ensure consistent behavior and easy tuning.

Usage:
    from victor.config.timeouts import Timeouts

    # HTTP requests
    async with httpx.AsyncClient(timeout=Timeouts.HTTP_DEFAULT) as client:
        ...

    # Process execution
    await asyncio.wait_for(process.communicate(), timeout=Timeouts.BASH_DEFAULT)

    # MCP communication
    await asyncio.wait_for(response, timeout=Timeouts.MCP_RESPONSE)
"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TimeoutConfig:
    """Centralized timeout configuration.

    All values are in seconds unless otherwise noted.
    Environment variables can override defaults:
        VICTOR_TIMEOUT_HTTP_DEFAULT=30.0
        VICTOR_TIMEOUT_BASH_DEFAULT=120
        etc.
    """

    # =========================================================================
    # HTTP/Network Timeouts
    # =========================================================================

    # Default HTTP request timeout
    HTTP_DEFAULT: float = 30.0

    # Quick HTTP checks (health checks, availability probes)
    HTTP_QUICK_CHECK: float = 2.0

    # Longer timeout for slow endpoints (model info)
    HTTP_SLOW_ENDPOINT: float = 5.0

    # Web search/fetch operations
    HTTP_WEB_SEARCH: float = 15.0

    # Embedding model requests (can be slow for large models)
    HTTP_EMBEDDING: float = 120.0

    # LLM API calls (can be very slow for complex queries)
    HTTP_LLM_API: float = 300.0

    # =========================================================================
    # Process/Subprocess Timeouts
    # =========================================================================

    # Default bash command timeout
    BASH_DEFAULT: int = 60

    # Quick commands (git status, ls, etc.)
    BASH_QUICK: int = 30

    # Long-running commands (npm install, pip install)
    BASH_LONG_RUNNING: int = 300

    # Test execution timeout
    TEST_EXECUTION: int = 300

    # Docker operations
    DOCKER_QUICK: int = 5
    DOCKER_LOGS: int = 30
    DOCKER_LOGS_FOLLOW: int = 60
    DOCKER_BUILD: int = 300

    # Git operations
    GIT_DEFAULT: int = 30

    # =========================================================================
    # MCP (Model Context Protocol) Timeouts
    # =========================================================================

    # MCP response timeout
    MCP_RESPONSE: float = 30.0

    # MCP process termination timeout
    MCP_PROCESS_TERMINATE: float = 5.0

    # MCP process force kill timeout
    MCP_PROCESS_KILL: float = 2.0

    # MCP server idle timeout
    MCP_SERVER_IDLE: float = 300.0

    # =========================================================================
    # Provider/Circuit Breaker Timeouts
    # =========================================================================

    # Circuit breaker recovery timeout
    CIRCUIT_BREAKER_RECOVERY: float = 30.0

    # =========================================================================
    # Code Indexing Timeouts
    # =========================================================================

    # Codebase index build timeout (dynamic based on codebase size, this is max)
    INDEX_BUILD_MAX: float = 600.0  # 10 minutes for very large codebases

    # Codebase index integrity probe timeout (quick health check)
    INDEX_PROBE: float = 15.0  # Increased from 5s to reduce false rebuilds

    # Symbol embedding generation timeout (per batch)
    SYMBOL_EMBEDDING_BATCH: float = 30.0

    @classmethod
    def from_env(cls) -> "TimeoutConfig":
        """Create config with environment variable overrides.

        Environment variables follow the pattern VICTOR_TIMEOUT_{FIELD_NAME}.
        Example: VICTOR_TIMEOUT_HTTP_DEFAULT=60.0
        """

        def get_float(name: str, default: float) -> float:
            env_key = f"VICTOR_TIMEOUT_{name}"
            value = os.environ.get(env_key)
            if value is not None:
                try:
                    return float(value)
                except ValueError:
                    pass
            return default

        def get_int(name: str, default: int) -> int:
            env_key = f"VICTOR_TIMEOUT_{name}"
            value = os.environ.get(env_key)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    pass
            return default

        return cls(
            HTTP_DEFAULT=get_float("HTTP_DEFAULT", cls.HTTP_DEFAULT),
            HTTP_QUICK_CHECK=get_float("HTTP_QUICK_CHECK", cls.HTTP_QUICK_CHECK),
            HTTP_SLOW_ENDPOINT=get_float("HTTP_SLOW_ENDPOINT", cls.HTTP_SLOW_ENDPOINT),
            HTTP_WEB_SEARCH=get_float("HTTP_WEB_SEARCH", cls.HTTP_WEB_SEARCH),
            HTTP_EMBEDDING=get_float("HTTP_EMBEDDING", cls.HTTP_EMBEDDING),
            HTTP_LLM_API=get_float("HTTP_LLM_API", cls.HTTP_LLM_API),
            BASH_DEFAULT=get_int("BASH_DEFAULT", cls.BASH_DEFAULT),
            BASH_QUICK=get_int("BASH_QUICK", cls.BASH_QUICK),
            BASH_LONG_RUNNING=get_int("BASH_LONG_RUNNING", cls.BASH_LONG_RUNNING),
            TEST_EXECUTION=get_int("TEST_EXECUTION", cls.TEST_EXECUTION),
            DOCKER_QUICK=get_int("DOCKER_QUICK", cls.DOCKER_QUICK),
            DOCKER_LOGS=get_int("DOCKER_LOGS", cls.DOCKER_LOGS),
            DOCKER_LOGS_FOLLOW=get_int("DOCKER_LOGS_FOLLOW", cls.DOCKER_LOGS_FOLLOW),
            DOCKER_BUILD=get_int("DOCKER_BUILD", cls.DOCKER_BUILD),
            GIT_DEFAULT=get_int("GIT_DEFAULT", cls.GIT_DEFAULT),
            MCP_RESPONSE=get_float("MCP_RESPONSE", cls.MCP_RESPONSE),
            MCP_PROCESS_TERMINATE=get_float("MCP_PROCESS_TERMINATE", cls.MCP_PROCESS_TERMINATE),
            MCP_PROCESS_KILL=get_float("MCP_PROCESS_KILL", cls.MCP_PROCESS_KILL),
            MCP_SERVER_IDLE=get_float("MCP_SERVER_IDLE", cls.MCP_SERVER_IDLE),
            CIRCUIT_BREAKER_RECOVERY=get_float(
                "CIRCUIT_BREAKER_RECOVERY", cls.CIRCUIT_BREAKER_RECOVERY
            ),
            INDEX_BUILD_MAX=get_float("INDEX_BUILD_MAX", cls.INDEX_BUILD_MAX),
            INDEX_PROBE=get_float("INDEX_PROBE", cls.INDEX_PROBE),
            SYMBOL_EMBEDDING_BATCH=get_float("SYMBOL_EMBEDDING_BATCH", cls.SYMBOL_EMBEDDING_BATCH),
        )


# Default singleton instance with environment overrides
Timeouts = TimeoutConfig.from_env()


# Convenience aliases for common timeout categories
class HttpTimeouts:
    """HTTP-specific timeout constants."""

    DEFAULT = Timeouts.HTTP_DEFAULT
    QUICK = Timeouts.HTTP_QUICK_CHECK
    SLOW = Timeouts.HTTP_SLOW_ENDPOINT
    WEB_SEARCH = Timeouts.HTTP_WEB_SEARCH
    EMBEDDING = Timeouts.HTTP_EMBEDDING
    LLM_API = Timeouts.HTTP_LLM_API


class ProcessTimeouts:
    """Process execution timeout constants."""

    BASH_DEFAULT = Timeouts.BASH_DEFAULT
    BASH_QUICK = Timeouts.BASH_QUICK
    BASH_LONG = Timeouts.BASH_LONG_RUNNING
    TEST = Timeouts.TEST_EXECUTION
    GIT = Timeouts.GIT_DEFAULT


class DockerTimeouts:
    """Docker operation timeout constants."""

    QUICK = Timeouts.DOCKER_QUICK
    LOGS = Timeouts.DOCKER_LOGS
    LOGS_FOLLOW = Timeouts.DOCKER_LOGS_FOLLOW
    BUILD = Timeouts.DOCKER_BUILD


class McpTimeouts:
    """MCP protocol timeout constants."""

    RESPONSE = Timeouts.MCP_RESPONSE
    TERMINATE = Timeouts.MCP_PROCESS_TERMINATE
    KILL = Timeouts.MCP_PROCESS_KILL
    SERVER_IDLE = Timeouts.MCP_SERVER_IDLE


@dataclass(frozen=True)
class SubprocessResourceLimits:
    """Resource limits for tool subprocess execution.

    Applied via POSIX ``resource.setrlimit`` in a ``preexec_fn`` callback.
    On non-POSIX platforms these limits are silently skipped.

    Environment variables follow the pattern VICTOR_SUBPROCESS_{FIELD_NAME}.
    """

    max_memory_mb: int = 512
    max_cpu_seconds: int = 300
    max_output_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_file_descriptors: int = 1024

    @classmethod
    def from_env(cls) -> "SubprocessResourceLimits":
        """Create config with environment variable overrides."""

        def _int(name: str, default: int) -> int:
            val = os.environ.get(f"VICTOR_SUBPROCESS_{name}")
            if val is not None:
                try:
                    return int(val)
                except ValueError:
                    pass
            return default

        return cls(
            max_memory_mb=_int("MAX_MEMORY_MB", cls.max_memory_mb),
            max_cpu_seconds=_int("MAX_CPU_SECONDS", cls.max_cpu_seconds),
            max_output_bytes=_int("MAX_OUTPUT_BYTES", cls.max_output_bytes),
            max_file_descriptors=_int("MAX_FILE_DESCRIPTORS", cls.max_file_descriptors),
        )
