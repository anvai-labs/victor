# ADR-006: Provider Integration Improvements for Non-Interactive Environments

## Metadata

- **Status**: Proposed
- **Date**: 2026-02-28
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 001 (Agent Orchestration), 002 (State Management)

## Context

### Problem Statement

The interviewer application (a product built on Victor) encountered critical integration issues when using cloud provider scoring jobs in non-interactive environments:

1. **Keychain Blocking**: macOS keychain access requests block background jobs, causing timeouts
2. **Silent Failures**: Provider initialization failures are not surfaced to users
3. **No Debug Visibility**: Lack of logging makes troubleshooting API key resolution nearly impossible
4. **Generic Timeouts**: All provider errors manifest as generic timeouts without actionable error messages
5. **No Pre-flight Checks**: No way to verify provider configuration before submitting long-running jobs

### User Impact

**Primary Affected Users**:
- Backend/daemon applications using Victor as a library
- CI/CD pipelines running automated scoring
- Containerized deployments without interactive access
- Applications using Victor's `ManagedProviderFactory` without explicit API key passing

**Symptoms**:
- Jobs hang for 195 seconds (timeout) then fail
- Users see generic `worker_timeout_exceeded` errors
- No indication that keychain access was the root cause
- Environment variables work but users don't know they're the solution

## Decision

We will implement a comprehensive provider integration improvement with four phases:

### Phase 1: Unified API Key Resolution (CRITICAL)

**Changes**:
1. Create `UnifiedApiKeyResolver` that all providers MUST use
2. Make environment variables the DEFAULT (not keychain)
3. Add non-interactive mode flag for daemon processes
4. Add pre-flight checks with actionable warnings

**Resolution Order (NEW)**:
```
1. Explicit api_key parameter (highest priority)
2. Environment variable (NEW DEFAULT for non-interactive)
3. VICTOR_NONINTERACTIVE env var → skip keychain
4. System keyring (only when VICTOR_NONINTERACTIVE=false/undefined)
5. Config file (~/.victor/api_keys.yaml)
```

**Key Insight**: Environment variables are the standard for non-interactive workloads (Docker, Kubernetes, CI/CD). Keychain is great for CLI tools but terrible for daemons.

### Phase 2: Structured Provider Logging (CRITICAL)

**Changes**:
1. Create `ProviderLogger` with structured logging events
2. Log all API key resolution attempts with source (env/keyring/file)
3. Log provider initialization with configuration (excluding secrets)
4. Add timing instrumentation for API calls
5. Add TRACE level for verbose debugging

**Log Events**:
```
PROVIDER_INIT: {provider, model, has_api_key, key_source, non_interactive}
API_KEY_RESOLUTION: {provider, source, success, latency_ms}
API_CALL_START: {provider, model, endpoint, timeout}
API_CALL_SUCCESS: {provider, model, latency_ms, tokens}
API_CALL_ERROR: {provider, model, error_type, error_code, retryable}
```

### Phase 3: Rich Error Types & Messages (HIGH)

**Changes**:
1. Create provider-specific exception hierarchy
2. Add `ProviderInitializationError` for config issues
3. Add `APIKeyNotFoundError` with actionable suggestions
4. Add `ProviderAuthenticationError` for invalid keys
5. Add `ProviderRateLimitError` with retry_after
6. Add `ProviderNetworkError` for connectivity issues

**Error Messages**:
```python
# Before
ProviderError: Request timed out after 195s

# After
APIKeyNotFoundError:
    DeepSeek API key not found. Tried the following sources:
    1. Explicit api_key parameter: not provided
    2. Environment variable DEEPSEEK_API_KEY: not set
    3. System keyring: access requires user interaction

    Solutions:
    • Set DEEPSEEK_API_KEY environment variable (recommended for servers)
    • Run 'victor keys set deepseek --keyring' for interactive use
    • Pass api_key parameter explicitly to ManagedProviderFactory.create()

    Context: Running in non-interactive mode (VICTOR_NONINTERACTIVE=true)
    Provider: deepseek
    Model: deepseek-chat
```

### Phase 4: Provider Health Check API (MEDIUM)

**Changes**:
1. Add `ProviderHealthCheck` class
2. Test API key resolution without making API calls
3. Optional connectivity test (configurable)
4. Expose via CLI: `victor providers check deepseek`

**CLI Output**:
```
$ victor providers check deepseek

✓ Provider registered
✓ API key found (source: environment variable)
✓ API key format valid (sk-*)
? Connectivity check: [Skip, use --connectivity to enable]

Provider: deepseek
Model: deepseek-chat
Key Source: DEEPSEEK_API_KEY environment variable
Status: HEALTHY
```

## Architecture

### Component Design

```python
# victor/providers/resolution.py

class UnifiedApiKeyResolver:
    """Centralized API key resolution with non-interactive support."""

    def __init__(self, non_interactive: bool = None):
        # Detect from environment or parameter
        self.non_interactive = (
            non_interactive or
            os.environ.get("VICTOR_NONINTERACTIVE", "").lower() == "true"
        )

    def get_api_key(
        self,
        provider: str,
        explicit_key: Optional[str] = None,
    ) -> APIKeyResult:
        """
        Resolve API key with full attribution.

        Returns:
            APIKeyResult with:
                - key: The API key (or None)
                - source: Where it came from
                - source_detail: Specific source (e.g., "DEEPSEEK_API_KEY env var")
                - interactive_required: Whether user interaction is needed
                - confidence: How confident we are (high/medium/low)
        """

class APIKeyNotFoundError(ProviderError):
    """API key not found with actionable suggestions."""

    def __init__(
        self,
        provider: str,
        sources_attempted: List[KeySource],
        non_interactive: bool,
    ):
        self.provider = provider
        self.sources_attempted = sources_attempted
        self.non_interactive = non_interactive

    def __str__(self) -> str:
        return self._format_actionable_message()

    def _format_actionable_message(self) -> str:
        """Generate user-friendly error with solutions."""
        lines = [
            f"{self.provider.upper()} API key not found. "
            f"Tried {len(self.sources_attempted)} sources:"
        ]
        for i, source in enumerate(self.sources_attempted, 1):
            status = "✓" if source.found else "✗"
            lines.append(f"{i}. {status} {source.description}")

        lines.append("\nSolutions:")
        if self.non_interactive:
            lines.append(
                f"• Set {self.provider.upper()}_API_KEY environment variable "
                "(recommended for servers/containers)"
            )
        else:
            lines.append(
                f"• Run: victor keys set {self.provider} --keyring"
            )
        lines.append(
            f"• Pass api_key parameter to ManagedProviderFactory.create()"
        )

        return "\n".join(lines)


# victor/providers/logging.py

class ProviderLogger:
    """Structured logging for provider operations."""

    def __init__(self, provider_name: str, logger_name: str):
        self.provider = provider_name
        self.logger = logging.getLogger(logger_name)
        self._initialize_structured_logging()

    def log_provider_init(
        self,
        model: str,
        key_source: Optional[str],
        non_interactive: bool,
        config: Dict[str, Any],
    ):
        self.logger.info(
            "PROVIDER_INIT",
            extra={
                "provider": self.provider,
                "model": model,
                "key_source": key_source,
                "non_interactive": non_interactive,
                "config": self._sanitize_config(config),
            }
        )

    def log_api_call(
        self,
        endpoint: str,
        model: str,
        latency_ms: float,
        tokens: Optional[int] = None,
        error: Optional[Exception] = None,
    ):
        self.logger.info(
            "API_CALL",
            extra={
                "provider": self.provider,
                "endpoint": endpoint,
                "model": model,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "error": str(error) if error else None,
            }
        )


# victor/providers/health.py

class ProviderHealthChecker:
    """Pre-flight health checks for provider configuration."""

    async def check_provider(
        self,
        provider: str,
        model: str,
        check_connectivity: bool = False,
        timeout: float = 5.0,
    ) -> ProviderHealthResult:
        """
        Check if provider is properly configured.

        Args:
            provider: Provider name
            model: Model to check
            check_connectivity: Make actual API call (slower but thorough)
            timeout: Timeout for connectivity check

        Returns:
            ProviderHealthResult with status and actionable issues
        """
```

### Provider Updates

All cloud providers (OpenAI, Anthropic, DeepSeek, etc.) will be updated:

```python
# victor/providers/deepseek_provider.py (UPDATED)

class DeepSeekProvider(BaseProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        # NEW: Use UnifiedApiKeyResolver
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        result = resolver.get_api_key("deepseek", explicit_key=api_key)

        if result.key is None:
            raise APIKeyNotFoundError(
                provider="deepseek",
                sources_attempted=result.sources_attempted,
                non_interactive=result.non_interactive,
            )

        # NEW: Use ProviderLogger
        self._logger = ProviderLogger("deepseek", __name__)
        self._logger.log_provider_init(
            model="deepseek-chat",  # Will be set on chat()
            key_source=result.source_detail,
            non_interactive=result.non_interactive,
            config={"base_url": base_url, "timeout": timeout},
        )

        self._api_key = result.key
        self._base_url = base_url
        self._timeout = timeout
        # ... rest of init
```

## Migration Plan

### Phase 1: Core Infrastructure (Week 1-2)

1. Create `UnifiedApiKeyResolver` in `victor/providers/resolution.py`
2. Create `APIKeyNotFoundError` and structured error types
3. Create `ProviderLogger` in `victor/providers/logging.py`
4. Update `victor/config/api_keys.py` to add `VICTOR_NONINTERACTIVE` support
5. Add unit tests for resolver

### Phase 2: Provider Updates (Week 3-4)

1. Update `DeepSeekProvider` to use new infrastructure
2. Update `AnthropicProvider` to use new infrastructure
3. Update `OpenAIProvider` to use new infrastructure
4. Add integration tests for error messages
5. Update documentation

### Phase 3: Health Check API (Week 5)

1. Create `ProviderHealthChecker` in `victor/providers/health.py`
2. Add `victor providers check` CLI command
3. Add health check to ManagedProviderFactory
4. Document pre-flight checks

### Phase 4: interviewer Integration (Week 6)

1. Update interviewer app to use new error types
2. Add provider pre-flight checks before scoring jobs
3. Add structured logging to interviewer backend
4. Update interviewer documentation

## Consequences

### Positive

1. **Better DX**: Users get actionable error messages instead of generic timeouts
2. **Production Ready**: Non-interactive mode works correctly for daemons/containers
3. **Debuggable**: Structured logging makes troubleshooting easy
4. **Testable**: Pre-flight checks catch config issues before long jobs
5. **Standard**: Environment variables follow 12-factor app patterns

### Negative

1. **Breaking Change**: Error messages change (but for the better)
2. **Dependency**: All providers must use UnifiedApiKeyResolver
3. **Migration**: Existing code needs updates for non-interactive mode

### Risks

1. **Incomplete Provider Updates**: Some providers might miss the migration
   - Mitigation: Add lint rule to check for direct `os.environ.get()` calls
2. **Logging Overhead**: Structured logging adds overhead
   - Mitigation: Make log level configurable, default to INFO
3. **Backward Compatibility**: Existing code might rely on keychain by default
   - Mitigation: Keep keychain as default for interactive mode

## Alternatives Considered

### Alternative 1: Keep Keychain as Default

**Rejected**: Keychain is fundamentally incompatible with non-interactive environments. Keeping it as default would cause the same issues we're trying to fix.

### Alternative 2: Require Explicit API Keys

**Rejected**: Forces all users to change code. Environment variables as default provides better UX.

### Alternative 3: Separate Daemon-Only Providers

**Rejected**: Would duplicate provider code. Better to have a single codepath with mode detection.

## Implementation Notes

### Environment Variable Detection

```python
# In UnifiedApiKeyResolver
def _detect_non_interactive(self) -> bool:
    """Detect if running in non-interactive environment."""
    # Explicit env var
    if os.environ.get("VICTOR_NONINTERACTIVE", "").lower() == "true":
        return True

    # Heuristics for common non-interactive environments
    if os.environ.get("CI"):  # GitHub Actions, GitLab CI, etc.
        return True
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    if os.environ.get("container"):  # Docker
        return True

    # Check if we have a TTY (Unix)
    try:
        import sys
        return not sys.stdin.isatty()
    except Exception:
        return False
```

### Interviewer Integration Example

```python
# interviewer/backend/scoring/local_llm_scorer.py (UPDATED)

async def score_with_local_model(
    scoring_request: dict[str, Any],
    *,
    model_cfg: LocalModelConfig | None = None,
):
    """Score with provider pre-flight check."""

    cfg = model_cfg or LocalModelConfig()

    # NEW: Pre-flight check
    health = await ProviderHealthChecker().check_provider(
        provider=cfg.provider,
        model=cfg.model,
        check_connectivity=False,  # Fast check only
    )

    if not health.healthy:
        # Rich error with actionable suggestions
        raise ProviderConfigurationError(
            f"Provider {cfg.provider} is not configured correctly.\n"
            f"{health.error_message}\n\n"
            f"Please fix before submitting scoring jobs."
        )

    # Proceed with confidence
    client = VictorChatClient(cfg)
    result = await client.complete(system_prompt, user_prompt)
    # ...
```

## References

- Technical Debt: `/Users/vijaysingh/code/codingagent/docs/tech-debt/victor-cloud-provider-timeout-fix.md`
- Interviewer App: `/Users/vijaysingh/code/interviewer`
- API Keys Module: `victor/config/api_keys.py`
- Provider Factory: `victor/providers/factory.py`
