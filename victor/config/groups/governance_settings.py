"""Governance policy configuration.

Settings for the governance policy engine (``victor.framework.policies``): a
declarative ALLOW / DENY / ASK layer over tool execution. Disabled by default;
enabling also requires the ``USE_POLICY_ENGINE`` feature flag.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class GovernanceSettings(BaseModel):
    """Governance / policy-engine settings.

    Controls cost budgets, tool approval gates, and per-session tool-call caps.
    """

    # ==========================================================================
    # Master switch
    # ==========================================================================
    # Enable the policy engine. Even when True, the USE_POLICY_ENGINE feature
    # flag must also be on for the middleware to be wired in.
    enabled: bool = False

    # ==========================================================================
    # Cost budget
    # ==========================================================================
    # Hard cap on cumulative session cost (USD). 0 disables the hard gate.
    cost_budget_usd: float = 0.0
    # Soft checkpoints (USD); each triggers a one-time approval prompt as the
    # running cost crosses it.
    cost_ask_thresholds_usd: List[float] = Field(default_factory=list)
    # Model-name substrings the hard cost gate applies to. Empty = all models.
    expensive_models: List[str] = Field(default_factory=lambda: ["opus", "fable", "gpt-5.5"])

    # ==========================================================================
    # Tool approval gate
    # ==========================================================================
    # Tool names that require human approval before running.
    ask_on_tools: List[str] = Field(default_factory=list)
    # Tool names hard-blocked outright (DENY, no approval bypass).
    deny_tools: List[str] = Field(default_factory=list)
    # Allowlist: when non-empty, any tool NOT listed is blocked.
    allow_tools: List[str] = Field(default_factory=list)

    # ==========================================================================
    # Tool-call cap
    # ==========================================================================
    # Maximum tool calls per session. 0 disables the cap.
    max_tool_calls_per_session: int = 0

    # ==========================================================================
    # Message phases (REQUEST: user message pre-LLM; RESPONSE: assistant output)
    # ==========================================================================
    # Regex patterns redacted from BOTH the user message and the assistant
    # output (matches replaced with redact_placeholder). Empty disables.
    redact_patterns: List[str] = Field(default_factory=list)
    # Replacement text for redacted matches.
    redact_placeholder: str = "[REDACTED]"
    # Regex patterns that hard-block (DENY) the user message before the LLM call.
    block_request_patterns: List[str] = Field(default_factory=list)
    # Regex patterns that hard-block (DENY) the assistant's final output.
    block_response_patterns: List[str] = Field(default_factory=list)

    # ==========================================================================
    # Approval fallback
    # ==========================================================================
    # Outcome of an ASK verdict when no approval handler is wired:
    # "deny" (fail safe, default) or "allow".
    ask_fallback: str = "deny"

    # When True (and stdin is an interactive TTY), wire the built-in console
    # approval handler so ASK verdicts prompt the user. Otherwise ASK resolves
    # via ask_fallback. Off by default so non-interactive/headless runs stay
    # deterministic.
    interactive_approval: bool = False

    # FEP-0029: durable pause. When True, an ASK verdict on a single-agent turn
    # raises ApprovalPause (parking the turn as a resumable ``awaiting_approval``
    # result) instead of blocking the inline modal. Off by default; intended on
    # for headless/API surfaces so they can defer approval rather than hang.
    durable: bool = False

    @field_validator("cost_budget_usd")
    @classmethod
    def validate_cost_budget(cls, v: float) -> float:
        """Cost budget must not be negative."""
        if v < 0:
            raise ValueError("cost_budget_usd must be >= 0")
        return v

    @field_validator("max_tool_calls_per_session")
    @classmethod
    def validate_max_tool_calls(cls, v: int) -> int:
        """Tool-call cap must not be negative."""
        if v < 0:
            raise ValueError("max_tool_calls_per_session must be >= 0")
        return v

    @field_validator("ask_fallback")
    @classmethod
    def validate_ask_fallback(cls, v: str) -> str:
        """ask_fallback must be 'deny' or 'allow'."""
        normalized = v.lower().strip()
        if normalized not in {"deny", "allow"}:
            raise ValueError("ask_fallback must be 'deny' or 'allow'")
        return normalized
