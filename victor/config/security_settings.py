"""Server security, sandboxing, and approval settings."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, SecretStr


class SecuritySettings(BaseModel):
    """Server security, sandboxing, and approval settings."""

    airgapped_mode: bool = False
    server_api_key: Optional[SecretStr] = None
    server_session_secret: Optional[SecretStr] = None
    server_max_sessions: int = 100
    server_max_message_bytes: int = 32768
    server_session_ttl_seconds: int = 86400
    render_max_payload_bytes: int = 20000
    render_timeout_seconds: int = 10
    render_max_concurrency: int = 2
    code_executor_network_disabled: bool = True
    code_executor_memory_limit: Optional[str] = "512m"
    code_executor_cpu_shares: Optional[int] = 256
    code_executor_image: str = "python:3.11-slim"
    code_executor_idle_ttl_seconds: int = 900
    write_approval_mode: str = "risky_only"
    headless_mode: bool = False
    dry_run_mode: bool = False
    auto_approve_safe: bool = False
    max_file_changes: Optional[int] = None
    security_dependency_scan: bool = False
    security_iac_scan: bool = False
    docker_allow_dangerous_operations: bool = False
