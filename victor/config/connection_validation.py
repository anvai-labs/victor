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

"""Connection testing and validation for provider accounts.

This module provides utilities to:
- Test provider connections
- Validate API keys
- Check model availability
- Diagnose configuration issues

Used by the auth setup wizard to verify credentials before saving.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.config.accounts import ProviderAccount, AuthConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Status
# =============================================================================


class ValidationStatus(Enum):
    """Status of a validation check."""

    SUCCESS = "success"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ConnectionValidationResult:
    """Result of a validation check."""

    status: ValidationStatus
    message: str
    details: Optional[str] = None
    latency_ms: Optional[int] = None

    def __bool__(self) -> bool:
        """Return True if validation succeeded."""
        return self.status == ValidationStatus.SUCCESS


@dataclass
class ConnectionTestResult:
    """Result of a connection test."""

    success: bool
    account_name: str
    provider: str
    model: str
    validations: List[ConnectionValidationResult] = field(default_factory=list)

    # Overall test info
    latency_ms: Optional[int] = None
    error: Optional[str] = None

    def add_validation(self, result: ConnectionValidationResult) -> None:
        """Add a validation result."""
        self.validations.append(result)

    def get_summary(self) -> str:
        """Get a summary of the test results."""
        if self.success:
            return f"✓ Connection to {self.provider}/{self.model} successful"
        else:
            return f"✗ Connection failed: {self.error or 'Unknown error'}"

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(v.status == ValidationStatus.WARNING for v in self.validations)


# =============================================================================
# Connection Validator
# =============================================================================


class ConnectionValidator:
    """Validate provider connections and credentials.

    Usage:
        validator = ConnectionValidator()

        # Test an account
        result = await validator.test_account(account)
        if result.success:
            print("Connection successful!")

        # Test synchronously
        result = validator.test_account_sync(account)
    """

    # Provider-specific validation endpoints
    VALIDATION_ENDPOINTS: Dict[str, str] = {
        "anthropic": "https://api.anthropic.com/v1/messages",
        "openai": "https://api.openai.com/v1/models",
        "google": "https://generativelanguage.googleapis.com/v1beta/models",
        "xai": "https://api.x.ai/v1/models",
        "moonshot": "https://api.moonshot.cn/v1/models",
        "deepseek": "https://api.deepseek.com/v1/models",
        "zai": "https://api.z.ai/api/paas/v4/models",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
        "groqcloud": "https://api.groq.com/openai/v1/models",
        "cerebras": "https://api.cerebras.ai/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
        "together": "https://api.together.xyz/v1/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
        "fireworks": "https://api.fireworks.ai/inference/v1/models",
    }

    # Local provider check URLs
    LOCAL_ENDPOINTS: Dict[str, Tuple[str, str]] = {
        "ollama": ("http://127.0.0.1:11434", "GET"),
        "lmstudio": ("http://127.0.0.1:1234", "GET"),
        "vllm": ("http://127.0.0.1:8000", "GET"),
    }

    def __init__(self, timeout: int = 10):
        """Initialize validator.

        Args:
            timeout: Request timeout in seconds
        """
        self._timeout = timeout

    async def test_account(self, account: ProviderAccount) -> ConnectionTestResult:
        """Test a provider account connection (async).

        Args:
            account: Provider account to test

        Returns:
            ConnectionTestResult with details
        """
        result = ConnectionTestResult(
            success=False,
            account_name=account.name,
            provider=account.provider,
            model=account.model,
        )

        # Check if local provider
        if account.is_local():
            return await self._test_local_provider(account, result)

        # Check authentication
        auth_result = await self._validate_auth(account)
        result.add_validation(auth_result)

        if not auth_result:
            result.error = f"Authentication failed: {auth_result.message}"
            return result

        # Test API endpoint
        endpoint_result = await self._test_endpoint(account)
        result.add_validation(endpoint_result)

        if not endpoint_result:
            result.error = f"Endpoint test failed: {endpoint_result.message}"
            return result

        # Validate model availability
        model_result = await self._validate_model(account)
        result.add_validation(model_result)

        if not model_result and model_result.status != ValidationStatus.WARNING:
            result.error = f"Model validation failed: {model_result.message}"
            return result

        result.success = True
        return result

    def test_account_sync(self, account: ProviderAccount) -> ConnectionTestResult:
        """Test a provider account connection (synchronous wrapper).

        Args:
            account: Provider account to test

        Returns:
            ConnectionTestResult with details
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.test_account(account))

    # ========================================================================
    # Private helper methods
    # ========================================================================

    async def _test_local_provider(
        self,
        account: ProviderAccount,
        result: ConnectionTestResult,
    ) -> ConnectionTestResult:
        """Test a local provider connection."""
        if account.provider not in self.LOCAL_ENDPOINTS:
            result.error = f"Unknown local provider: {account.provider}"
            return result

        url, method = self.LOCAL_ENDPOINTS[account.provider]

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self._timeout) as response:
                    if response.status < 500:
                        result.success = True
                        result.add_validation(
                            ConnectionValidationResult(
                                status=ValidationStatus.SUCCESS,
                                message=f"{account.provider.capitalize()} is running",
                            )
                        )
                    else:
                        result.error = f"{account.provider} returned status {response.status}"
        except asyncio.TimeoutError:
            result.error = f"Connection to {account.provider} timed out"
        except Exception as e:
            result.error = f"Failed to connect to {account.provider}: {e}"

        return result

    async def _validate_auth(self, account: ProviderAccount) -> ConnectionValidationResult:
        """Validate authentication configuration."""
        if account.auth.method == "none":
            return ConnectionValidationResult(
                status=ValidationStatus.SUCCESS,
                message="No authentication required",
            )

        if account.auth.method == "oauth":
            # Check for OAuth client_id
            client_id = self._get_oauth_client_id(account.provider)
            if client_id:
                return ConnectionValidationResult(
                    status=ValidationStatus.SUCCESS,
                    message="OAuth client_id configured",
                )
            else:
                return ConnectionValidationResult(
                    status=ValidationStatus.FAILED,
                    message="OAuth client_id not found in keyring",
                )

        # API key authentication
        api_key = self._get_api_key(account)

        if not api_key:
            return ConnectionValidationResult(
                status=ValidationStatus.FAILED,
                message="API key not found",
                details="Checked keyring and environment variables",
            )

        # Validate API key format
        if not self._validate_api_key_format(account.provider, api_key):
            return ConnectionValidationResult(
                status=ValidationStatus.WARNING,
                message="API key format may be invalid",
                details="Key doesn't match expected format for this provider",
            )

        return ConnectionValidationResult(
            status=ValidationStatus.SUCCESS,
            message="API key found",
        )

    async def _test_endpoint(self, account: ProviderAccount) -> ConnectionValidationResult:
        """Test the provider API endpoint."""
        import time
        import aiohttp

        endpoint = account.endpoint or self._resolve_endpoint(account)

        if not endpoint:
            return ConnectionValidationResult(
                status=ValidationStatus.WARNING,
                message="No endpoint configured",
                details="Using provider default",
            )

        # Build headers
        headers = self._build_headers(account)

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # For most providers, we can make a simple request
                # Use HEAD or GET with minimal data
                async with session.head(
                    endpoint,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    latency = int((time.time() - start_time) * 1000)

                    if response.status in (200, 401, 403):
                        # 200 = success, 401/403 = endpoint reachable but auth failed
                        return ConnectionValidationResult(
                            status=ValidationStatus.SUCCESS,
                            message="Endpoint reachable",
                            latency_ms=latency,
                        )
                    else:
                        return ConnectionValidationResult(
                            status=ValidationStatus.WARNING,
                            message=f"Endpoint returned status {response.status}",
                            latency_ms=latency,
                        )

        except asyncio.TimeoutError:
            return ConnectionValidationResult(
                status=ValidationStatus.FAILED,
                message="Connection timed out",
            )
        except Exception as e:
            return ConnectionValidationResult(
                status=ValidationStatus.FAILED,
                message=f"Connection failed: {e}",
            )

    async def _validate_model(self, account: ProviderAccount) -> ConnectionValidationResult:
        """Validate that the model is available."""
        # This is a simplified check - in reality, we'd query the provider
        # for available models. For now, we just do basic format validation.

        model = account.model

        # Check model format
        if not model or model == "default":
            return ConnectionValidationResult(
                status=ValidationStatus.WARNING,
                message="No specific model selected",
                details="Using provider default",
            )

        # Check for model suffix (endpoint variant)
        if ":" in model:
            base_model, variant = model.rsplit(":", 1)
            if account.provider == "zai":
                valid_variants = {"coding", "standard", "china", "anthropic"}
                if variant not in valid_variants:
                    return ConnectionValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"Unknown model variant: {variant}",
                        details=f"Valid variants for {account.provider}: {', '.join(valid_variants)}",
                    )

        return ConnectionValidationResult(
            status=ValidationStatus.SUCCESS,
            message=f"Model '{model}' selected",
        )

    def _get_api_key(self, account: ProviderAccount) -> Optional[str]:
        """Get API key from appropriate source."""
        import os

        if account.auth.source == "sentinelpass":
            try:
                from victor.providers.resolution import _get_key_from_sentinelpass

                return _get_key_from_sentinelpass(
                    account.auth.value or account.provider,
                    force=True,
                )
            except ImportError:
                return None

        # Check environment variable first
        env_var = self._get_provider_env_var(account.provider)
        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key

        # Check explicit value
        if account.auth.value:
            return account.auth.value

        # Check keyring
        try:
            from victor.config.api_keys import _get_key_from_keyring

            return _get_key_from_keyring(account.provider)
        except ImportError:
            pass

        return None

    def _resolve_endpoint(self, account: ProviderAccount) -> Optional[str]:
        """Resolve the correct endpoint for a provider account.

        OAuth accounts use different endpoints (e.g., Codex bridge for OpenAI).
        """
        if account.auth.method == "oauth":
            oauth_endpoints = {
                "openai": "https://chatgpt.com/backend-api/codex/models",
                "qwen": "https://portal.qwen.ai/v1/models",
            }
            return oauth_endpoints.get(account.provider)
        return self.VALIDATION_ENDPOINTS.get(account.provider)

    def _get_oauth_client_id(self, provider: str) -> Optional[str]:
        """Get OAuth client_id using the full resolution chain.

        Checks: environment variable -> keyring -> well-known public client IDs.
        """
        try:
            from victor.providers.oauth_manager import _get_oauth_client_id

            return _get_oauth_client_id(provider)
        except (ImportError, ValueError):
            return None

    def _get_provider_env_var(self, provider: str) -> Optional[str]:
        """Get environment variable name for provider."""
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "zai": "ZAI_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "groqcloud": "GROQCLOUD_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
        }
        return env_vars.get(provider)

    def _validate_api_key_format(self, provider: str, api_key: str) -> bool:
        """Validate API key format for provider."""
        if not api_key:
            return False

        # Basic format checks
        if provider == "anthropic":
            return api_key.startswith("sk-ant-")
        elif provider == "openai":
            return api_key.startswith("sk-")
        elif provider in ("google", "qwen"):
            return len(api_key) > 20  # Simplified check
        else:
            # Generic check: at least 20 characters
            return len(api_key) >= 20

    def _build_headers(self, account: ProviderAccount) -> Dict[str, str]:
        """Build HTTP headers for API request."""
        headers = {
            "User-Agent": "Victor/1.0",
        }

        # Add auth header
        if account.auth.method == "api_key":
            api_key = self._get_api_key(account)
            if api_key:
                if account.provider == "anthropic":
                    headers["x-api-key"] = api_key
                else:
                    headers["Authorization"] = f"Bearer {api_key}"
        elif account.auth.method == "oauth":
            # Use cached OAuth token if available (don't trigger login)
            token = self._get_cached_oauth_token(account.provider)
            if token:
                headers["Authorization"] = f"Bearer {token}"
                if account.provider == "openai":
                    headers["originator"] = "victor"

        return headers

    def _get_cached_oauth_token(self, provider: str) -> Optional[str]:
        """Get cached OAuth access token without triggering login."""
        try:
            from pathlib import Path

            from victor.config.secure_paths import get_victor_dir

            token_file = get_victor_dir() / "oauth_tokens.yaml"
        except ImportError:
            token_file = Path.home() / ".victor" / "oauth_tokens.yaml"

        if not token_file.exists():
            return None

        try:
            import yaml

            with open(token_file) as f:
                all_tokens = yaml.safe_load(f) or {}
            data = all_tokens.get(provider)
            if data and data.get("access_token"):
                return data["access_token"]
        except Exception:
            pass
        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_account_sync(account: ProviderAccount) -> ConnectionTestResult:
    """Validate an account connection (synchronous).

    Args:
        account: Provider account to validate

    Returns:
        ConnectionTestResult with details
    """
    validator = ConnectionValidator()
    return validator.test_account_sync(account)


async def validate_account_async(account: ProviderAccount) -> ConnectionTestResult:
    """Validate an account connection (async).

    Args:
        account: Provider account to validate

    Returns:
        ConnectionTestResult with details
    """
    validator = ConnectionValidator()
    return await validator.test_account(account)


def ping_provider(provider: str, endpoint: Optional[str] = None) -> ConnectionValidationResult:
    """Quick ping test for a provider.

    Args:
        provider: Provider name
        endpoint: Custom endpoint (optional)

    Returns:
        ConnectionValidationResult with status
    """
    import time

    from victor.config.accounts import ProviderAccount, AuthConfig

    account = ProviderAccount(
        name="ping-test",
        provider=provider,
        model="default",
        auth=AuthConfig(method="none" if provider in {"ollama", "lmstudio", "vllm"} else "api_key"),
        endpoint=endpoint,
    )

    validator = ConnectionValidator()
    result = validator.test_account_sync(account)

    return ConnectionValidationResult(
        status=ValidationStatus.SUCCESS if result.success else ValidationStatus.FAILED,
        message=result.get_summary(),
        latency_ms=result.latency_ms,
    )
