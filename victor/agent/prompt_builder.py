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

"""System prompt builder for Victor.

Builds provider-specific system prompts based on:
- Provider type (cloud vs local)
- Model capabilities (native tool calling vs fallback)
- Tool calling adapter hints
"""

import logging
from typing import TYPE_CHECKING, Optional, Set, List, Dict, Any

from victor.agent.prompt_section_texts import (
    ASI_TOOL_EFFECTIVENESS_GUIDANCE,
    COMPLETION_GUIDANCE,
    CONCISE_MODE_GUIDANCE,
    GROUNDING_RULES,
    GROUNDING_RULES_EXTENDED,
    HEADLESS_MODE_GUIDANCE,
    LARGE_FILE_PAGINATION_GUIDANCE,
    PARALLEL_READ_GUIDANCE,
)
from victor.agent.tool_calling import BaseToolCallingAdapter, ToolCallingCapabilities
from victor.agent.provider_tool_guidance import (
    get_tool_guidance_strategy,
    ToolGuidanceStrategy,
)
from victor.agent.prompt_section_registry import register_prompt_contributor_sections
from victor.agent.prompt_normalizer import get_prompt_normalizer
from victor.core.constants import DEFAULT_VERTICAL
from victor.core.verticals.protocols.prompt_provider import (
    collect_prompt_section_contributions,
)
from victor.framework.prompt_document import PromptBlock, PromptDocument
from victor.tools.core_tool_aliases import canonicalize_core_tool_name

if TYPE_CHECKING:
    from victor.agent.query_classifier import QueryClassification
    from victor.framework.enrichment import (
        PromptEnrichmentService,
        EnrichmentContext,
        EnrichedPrompt,
    )

logger = logging.getLogger(__name__)


# Provider classifications
CLOUD_PROVIDERS: Set[str] = {
    "anthropic",
    "openai",
    "google",
    "xai",
    "moonshot",
    "kimi",
    "deepseek",
}
LOCAL_PROVIDERS: Set[str] = {"ollama", "lmstudio", "vllm"}

# Task-type hints are now in vertical prompt contributors (E5 M3).
# Use get_task_type_hint(task_type, prompt_contributors=[...]) instead.


def get_task_type_hint(task_type: str, prompt_contributors: Optional[list] = None) -> str:
    """Get prompt hint for a specific task type.

    Hints come from vertical prompt contributors (the canonical source).
    Pass prompt_contributors from the vertical for full task hint support.

    Args:
        task_type: The detected task type (e.g., "create_simple", "edit")
        prompt_contributors: Optional list of PromptContributorProtocol implementations

    Returns:
        Task-specific prompt hint or empty string if not found
    """
    if prompt_contributors:
        for contributor in sorted(prompt_contributors, key=lambda c: c.get_priority()):
            hints = contributor.get_task_type_hints()
            if task_type.lower() in hints:
                task_hint = hints[task_type.lower()]
                # Handle TaskTypeHint objects or plain strings
                if hasattr(task_hint, "hint"):
                    hint_text = task_hint.hint
                else:
                    hint_text = str(task_hint)
                contributor_name = type(contributor).__name__
                logger.info(
                    "Applied vertical task hint: task_type=%s, contributor=%s",
                    task_type,
                    contributor_name,
                )
                return hint_text

    return ""


# Models with known good native tool calling support
NATIVE_TOOL_MODELS = [
    "qwen2.5",
    "qwen-2.5",
    "qwen3",
    "qwen-3",
    "llama-3.1",
    "llama3.1",
    "llama-3.2",
    "llama3.2",
    "llama-3.3",
    "llama3.3",
    "ministral",
    "mistral",
    "mixtral",
    "command-r",
    "firefunction",
    "hermes",
    "functionary",
]


class SystemPromptBuilder:
    """Builds system prompts tailored to provider and model capabilities.

    Different providers have different tool calling capabilities:
    - Cloud providers (Anthropic, OpenAI, Google, xAI): Robust native tool calling
    - vLLM: Production-grade OpenAI-compatible with tool parsers
    - LMStudio: OpenAI-compatible with Native vs Default mode
    - Ollama: Native tool_calls for Llama3.1+, Qwen2.5+, Mistral; fallback otherwise

    Vertical Integration:
    - Accepts prompt contributors from verticals via DI container
    - Merges vertical-specific task hints and system prompt sections
    - Falls back to deprecated hints for backward compatibility
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        tool_adapter: Optional[BaseToolCallingAdapter] = None,
        capabilities: Optional[ToolCallingCapabilities] = None,
        prompt_contributors: Optional[list] = None,
        tool_guidance_strategy: Optional[ToolGuidanceStrategy] = None,
        task_type: str = "medium",
        available_tools: Optional[List[str]] = None,
        enrichment_service: Optional["PromptEnrichmentService"] = None,
        vertical: Optional[str] = None,
        concise_mode: bool = False,
        headless_mode: bool = False,
        contextual_guidance: Optional[str] = None,
        query_classification: Optional["QueryClassification"] = None,
        mode_prompt_addition: str = "",
        provider_caches: bool = False,
        provider_has_kv_cache: bool = False,
        system_prompt_strategy: str = "static",
        llm_decision_service: Optional[Any] = None,
    ):
        """Initialize the prompt builder.

        Args:
            provider_name: Name of the provider (e.g., "ollama", "anthropic")
            model: Model name/identifier
            tool_adapter: Optional tool calling adapter for getting hints
            capabilities: Optional pre-computed capabilities
            prompt_contributors: Optional list of PromptContributorProtocol implementations
            tool_guidance_strategy: Optional provider-specific tool guidance strategy (GAP-5)
            task_type: Task complexity level (simple, medium, complex) for guidance
            available_tools: List of available tool names for guidance context
            enrichment_service: Optional prompt enrichment service for context injection
            vertical: Current vertical (coding, research, devops, data_analysis) for enrichment
            concise_mode: If True, adds guidance to produce brief, direct responses
            provider_caches: If True, provider supports API-level prompt caching (Anthropic).
                Full prompt is optimal (cached at 90% discount). If False, aggressively
                prune sections and skip MIPROv2 few-shots to save tokens.
            provider_has_kv_cache: If True, provider supports KV prefix caching (Ollama,
                LMStudio, etc.). Sections are reduced AND frozen at session start so the
                system prompt prefix stays byte-identical across turns.
            system_prompt_strategy: Strategy for system prompt management.
                - 'static': Freeze at session start for cache optimization (default)
                - 'dynamic': Rebuild per-turn based on context/tool calls
                - 'hybrid': Static for API providers, dynamic for local providers
            llm_decision_service: Optional LLM decision service for edge model decisions
        """
        # Handle both string and ProviderSettings object for provider_name
        # (backward compatibility with settings refactor)
        if provider_name and hasattr(provider_name, "default_provider"):
            # provider_name is a ProviderSettings object
            actual_provider_name = provider_name.default_provider
        else:
            # provider_name is a string or None
            actual_provider_name = provider_name

        self.provider_name = (actual_provider_name or "").lower()
        self.model = model or ""
        self.model_lower = self.model.lower()
        self.tool_adapter = tool_adapter
        self.capabilities = capabilities
        self.prompt_contributors = prompt_contributors or []
        if self.prompt_contributors:
            try:
                register_prompt_contributor_sections(self.prompt_contributors)
            except Exception:
                logger.debug("Failed to register contributor prompt sections", exc_info=True)
        self.task_type = task_type
        self.available_tools = available_tools or []
        self.stable_prompt_tools = list(self.available_tools)
        self.dynamic_prompt_tools: List[str] = []
        self.enrichment_service = enrichment_service
        self.vertical = vertical or DEFAULT_VERTICAL
        self.concise_mode = concise_mode
        self.headless_mode = headless_mode
        self.contextual_guidance = contextual_guidance
        self.query_classification = query_classification
        self.mode_prompt_addition = mode_prompt_addition.strip()
        self.provider_caches = provider_caches
        self.provider_has_kv_cache = provider_has_kv_cache
        self.system_prompt_strategy = system_prompt_strategy
        self._llm_decision_service = llm_decision_service or self._resolve_llm_decision_service()

        # Initialize tool guidance strategy (GAP-5: Provider-specific tool guidance)
        # Use provided strategy or auto-detect based on provider name
        if tool_guidance_strategy:
            self._tool_guidance = tool_guidance_strategy
        else:
            self._tool_guidance = get_tool_guidance_strategy(self.provider_name)

        # Cache merged task hints from vertical contributors
        self._merged_task_hints = None

        # Prompt caching for static mode
        self._cached_prompt: Optional[str] = None
        self._cache_key: Optional[str] = None
        self._evolved_content_resolver: Any = None

    @staticmethod
    def _resolve_llm_decision_service() -> Optional[Any]:
        """Resolve optional edge-decision service from the configured DI container."""
        try:
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )
            from victor.core.service_resolution import resolve_optional_service

            return resolve_optional_service(
                LLMDecisionServiceProtocol,
                legacy_key="llm_decision_service",
            )
        except Exception:
            return None

    def is_cloud_provider(self) -> bool:
        """Check if the provider is a cloud-based API with robust tool calling."""
        return self.provider_name in CLOUD_PROVIDERS

    def is_local_provider(self) -> bool:
        """Check if the provider is a local model (Ollama, LMStudio, vLLM)."""
        return self.provider_name in LOCAL_PROVIDERS

    def has_native_tool_support(self) -> bool:
        """Check if the model has known native tool calling support."""
        return any(pattern in self.model_lower for pattern in NATIVE_TOOL_MODELS)

    def get_merged_task_hints(self) -> dict:
        """Get merged task hints from vertical contributors.

        Returns:
            Dict of task type -> hint string, merged from all contributors
        """
        if self._merged_task_hints is not None:
            return self._merged_task_hints

        merged: dict = {}  # Populated by vertical contributors below

        # Override with vertical contributors (sorted by priority)
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            hints = contributor.get_task_type_hints()
            for task_type, task_hint in hints.items():
                # Extract hint string from TaskTypeHint objects
                if hasattr(task_hint, "hint"):
                    merged[task_type] = task_hint.hint
                else:
                    merged[task_type] = str(task_hint)

        self._merged_task_hints = merged
        return merged

    def get_vertical_grounding_rules(self) -> str:
        """Get grounding rules from vertical contributors with deduplication.

        Uses PromptNormalizer.deduplicate_sections() to remove duplicate
        grounding rules that may come from multiple contributors.

        Returns:
            Merged and deduplicated grounding rules from all contributors
        """
        if not self.prompt_contributors:
            return ""

        # Collect grounding rules from all contributors
        rules = []
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            grounding = contributor.get_grounding_rules()
            if grounding:
                rules.append(grounding)

        # Deduplicate sections using PromptNormalizer
        if rules:
            normalizer = get_prompt_normalizer()
            rules = normalizer.deduplicate_sections(rules)

        return "\n\n".join(rules) if rules else ""

    def get_vertical_system_prompt_sections(self) -> str:
        """Get system prompt sections from vertical contributors with deduplication.

        Uses PromptNormalizer.deduplicate_sections() to remove duplicate
        system prompt sections that may come from multiple contributors.

        Returns:
            Merged and deduplicated system prompt sections from all contributors
        """
        if not self.prompt_contributors:
            return ""

        # Collect sections from all contributors
        sections = []
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            for contribution in collect_prompt_section_contributions(contributor):
                if contribution.text:
                    sections.append(contribution.text)

        # Deduplicate sections using PromptNormalizer
        if sections:
            normalizer = get_prompt_normalizer()
            sections = normalizer.deduplicate_sections(sections)

        return "\n\n".join(sections) if sections else ""

    def get_provider_tool_guidance(self) -> str:
        """Get provider-specific tool usage guidance.

        This implements GAP-5: Provider-specific tool guidance (Strategy pattern).
        Different providers have different tool calling behaviors that benefit from
        tailored guidance. For example:
        - DeepSeek tends to over-explore with redundant tool calls
        - Grok handles tools efficiently with minimal redundancy
        - Ollama needs stricter guidance due to weaker tool calling

        Returns:
            Provider-specific tool guidance prompt or empty string if no special guidance needed
        """
        if not self._tool_guidance:
            return ""

        guidance = self._tool_guidance.get_guidance_prompt(
            task_type=self.task_type,
            available_tools=self.get_stable_prompt_tools(),
        )

        if guidance:
            logger.debug(
                f"Applied provider tool guidance for {self.provider_name}, "
                f"task_type={self.task_type}"
            )

        return guidance

    def get_task_guidance_text(self) -> str:
        """Expose task guidance for runtimes that inject it outside the system prompt."""
        return self._get_task_guidance_section()

    def get_contextual_guidance_text(self) -> str:
        """Expose per-turn contextual guidance for user-prefix injection.

        Mirrors :meth:`get_task_guidance_text`: lets KV-cache runtimes inject the
        guidance into the per-turn user message instead of the frozen system
        prompt, so the system prefix stays stable across turns (otherwise this
        per-turn content would be silently dropped once the prompt is frozen).
        """
        return self.contextual_guidance or ""

    def get_stable_prompt_tools(self) -> List[str]:
        """Return the stable tool set used in the system-prompt surface."""
        tools = getattr(self, "stable_prompt_tools", None)
        if tools is None:
            tools = self.available_tools
        return list(tools or [])

    def get_dynamic_prompt_tools(self) -> List[str]:
        """Return long-tail tools that should be hinted dynamically per turn."""
        return list(getattr(self, "dynamic_prompt_tools", []) or [])

    def get_dynamic_tool_guidance_text(
        self,
        relevant_tools: Optional[List[str]] = None,
        *,
        goals: Optional[List[str]] = None,
        current_intent: Optional[str] = None,
        selection_source: Optional[str] = None,
        tool_rationale: Optional[Dict[str, str]] = None,
    ) -> str:
        """Render dynamic tool hints for user-prefix injection.

        These hints are separate from provider tool schemas. They keep stable,
        frequently used tools in the system prompt while surfacing long-tail
        tools only when the current turn likely needs them.
        """
        source_tools = (
            relevant_tools if relevant_tools is not None else self.get_dynamic_prompt_tools()
        )
        normalized_tools = sorted(
            {canonicalize_core_tool_name(tool) for tool in source_tools if tool}
        )
        if not normalized_tools:
            return ""

        listed = normalized_tools[:6]
        more_count = max(0, len(normalized_tools) - len(listed))
        tool_list = ", ".join(listed)
        if more_count:
            tool_list = f"{tool_list}, and {more_count} more"

        guidance = (
            "DYNAMIC TOOL HINTS: The stable system prompt covers the core tools. "
            f"For this turn, the following less-common tools are relevant if needed: {tool_list}. "
        )

        if selection_source == "planned_tools":
            guidance += "These tools came from the current planned tool sequence. "
        elif selection_source == "keyword_selector":
            guidance += "These tools matched the current turn's explicit tool keywords. "

        if tool_rationale:
            rationale_parts = []
            for tool_name in listed:
                rationale = tool_rationale.get(tool_name)
                if rationale:
                    rationale_parts.append(f"{tool_name} ({rationale})")
            if rationale_parts:
                guidance += f"Relevant use hints: {'; '.join(rationale_parts)}. "

        if goals:
            compact_goals = [goal.strip() for goal in goals if goal and goal.strip()]
            if compact_goals:
                guidance += f"Current plan focus: {'; '.join(compact_goals[:3])}. "

        if current_intent:
            readable_intent = current_intent.replace("_", " ").replace("-", " ").strip()
            if readable_intent:
                guidance += f"Current intent guard: {readable_intent}. "

        guidance += "Only call these dynamic tools when the current task clearly requires them."
        return guidance

    def get_max_exploration_depth(self) -> int:
        """Get the maximum exploration depth for the current provider and task complexity.

        Returns:
            Maximum number of tool calls before synthesis checkpoint
        """
        if not self._tool_guidance:
            return 10  # Default

        return self._tool_guidance.get_max_exploration_depth(self.task_type)

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        """Check if tool calls should be consolidated based on history.

        Args:
            tool_history: List of recent tool calls with 'tool' and 'args' keys

        Returns:
            True if consolidation/synthesis is recommended
        """
        if not self._tool_guidance:
            return False

        return self._tool_guidance.should_consolidate_calls(tool_history)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        """Get synthesis checkpoint prompt if we've reached the threshold.

        Args:
            tool_count: Number of tool calls made so far

        Returns:
            Synthesis prompt or empty string if not at checkpoint
        """
        if not self._tool_guidance:
            return ""

        return self._tool_guidance.get_synthesis_checkpoint_prompt(tool_count)

    async def enrich_prompt(
        self,
        prompt: str,
        context: Optional["EnrichmentContext"] = None,
    ) -> "EnrichedPrompt":
        """Enrich a user prompt with contextual information.

        Uses the enrichment service to inject relevant context based on the
        vertical (coding, research, devops, data_analysis). This can include:
        - Knowledge graph symbols for coding tasks
        - Web search results for research tasks
        - Infrastructure context for devops tasks
        - Schema context for data analysis tasks

        Args:
            prompt: The user prompt to enrich
            context: Optional enrichment context with session/task metadata.
                     If not provided, a basic context will be created.

        Returns:
            EnrichedPrompt with the enriched prompt text and metadata.
            If enrichment is disabled or unavailable, returns the original prompt.
        """
        # Import here to avoid circular imports
        from victor.framework.enrichment import EnrichmentContext, EnrichedPrompt

        # If no enrichment service, return original prompt
        if not self.enrichment_service:
            logger.debug("No enrichment service available, returning original prompt")
            return EnrichedPrompt(
                original_prompt=prompt,
                enriched_prompt=prompt,
            )

        # Create default context if not provided
        if context is None:
            context = EnrichmentContext(
                task_type=self.task_type,
            )

        try:
            result = await self.enrichment_service.enrich(
                prompt=prompt,
                vertical=self.vertical,
                context=context,
            )
            logger.info(
                "Prompt enriched: vertical=%s, enrichments=%d, tokens_added=%d",
                self.vertical,
                result.enrichment_count,
                result.total_tokens_added,
            )
            return result
        except Exception as e:
            logger.warning("Prompt enrichment failed: %s", e)
            return EnrichedPrompt(
                original_prompt=prompt,
                enriched_prompt=prompt,
            )

    def _get_task_guidance_section(self) -> str:
        """Get task-specific guidance based on query classification."""
        if not self.query_classification:
            return ""

        from victor.agent.query_classifier import QueryType

        guidance_map = {
            QueryType.EXPLORATION: (
                "TASK GUIDANCE: Explore systematically. List directories before reading files. "
                "Map structure first. Build a mental model of the codebase before diving deep."
            ),
            QueryType.IMPLEMENTATION: (
                "TASK GUIDANCE: Plan before coding. Break into discrete steps. "
                "Test incrementally. Verify each step before proceeding."
            ),
            QueryType.DEBUGGING: (
                "TASK GUIDANCE: Focus on error messages and stack traces. "
                "Check recent changes first. Reproduce the issue before fixing."
            ),
            QueryType.REVIEW: (
                "TASK GUIDANCE: Examine for correctness, style, and security. "
                "Reference specific line numbers. Be thorough but constructive."
            ),
            QueryType.QUICK_QUESTION: (
                "TASK GUIDANCE: Answer directly and concisely. "
                "Only use tools if the answer requires checking the codebase."
            ),
        }
        return guidance_map.get(self.query_classification.query_type, "")

    def _get_mode_guidance_section(self) -> str:
        """Get current mode-specific guidance from the live runtime."""
        return self.mode_prompt_addition.strip()

    def _get_tool_constraint_section(self) -> str:
        """Get tool constraint section listing available tools."""
        normalized_tools = sorted(
            {canonicalize_core_tool_name(tool) for tool in self.get_stable_prompt_tools() if tool}
        )
        if not normalized_tools:
            return ""
        tool_list = ", ".join(normalized_tools)
        return (
            f"IMPORTANT: Only use tools from this list: {tool_list}. "
            "Do not attempt to call unlisted tools."
        )

    def build(self) -> str:
        """Build the system prompt.

        Uses adapter hints if available, otherwise falls back to
        provider-specific prompt construction. Includes provider-specific
        tool guidance (GAP-5) when available.

        The prompt is built in this order:
        1. Concise mode guidance (if enabled)
        2. Base prompt (provider-specific)
        3. Task-specific guidance (based on query classification)
        4. Tool constraint section (available tools list)
        5. Completion guidance (always included for deterministic task completion)
        6. Provider-specific tool guidance (GAP-5)

        System Prompt Strategy:
        - 'static': Cache prompt after first build, reuse for all turns (default)
        - 'dynamic': Rebuild every turn based on current context
        - 'hybrid': Static for API providers (cache benefit), dynamic for local providers

        Returns:
            System prompt string tailored to the provider/model
        """
        # Determine if we should use cached prompt or rebuild
        strategy = self._get_effective_strategy()

        if strategy == "static":
            # Return cached prompt if available
            if self._cached_prompt is not None:
                logger.debug(
                    "[SystemPrompt→Cache] Using cached prompt (strategy=static, len=%d)",
                    len(self._cached_prompt),
                )
                return self._cached_prompt

        # Build the prompt
        prompt = self.build_document().render()

        # Cache for static mode
        if strategy == "static":
            self._cached_prompt = prompt
            logger.debug(
                "[SystemPrompt→Cache] Cached prompt (strategy=static, len=%d)",
                len(prompt),
            )
        elif strategy == "dynamic":
            # Clear cache to ensure rebuild next time
            self._cached_prompt = None
            logger.debug(
                "[SystemPrompt→Dynamic] Rebuilt prompt (strategy=dynamic, len=%d)",
                len(prompt),
            )
        # hybrid: cache for API providers, no cache for local

        return prompt

    def build_document(self) -> PromptDocument:
        """Build the canonical system prompt document."""
        document = PromptDocument()

        if self.tool_adapter:
            base_prompt = self._build_with_adapter()
        else:
            base_prompt = self._build_for_provider()

        sections_to_include = self._get_active_sections()
        document.upsert(
            PromptBlock(
                name="base_prompt",
                content=base_prompt,
                priority=10,
                header="",
                kind="system_base",
            )
        )

        if "concise_mode" in sections_to_include and self.concise_mode:
            document.upsert(
                PromptBlock(
                    name="concise_mode",
                    content=self._resolve_optional_prompt_section(
                        "CONCISE_MODE_GUIDANCE",
                        CONCISE_MODE_GUIDANCE,
                    ),
                    priority=20,
                    header="",
                    kind="guidance",
                )
            )
            logger.debug("Concise mode enabled - added brevity guidance to prompt")

        if "headless_mode" in sections_to_include and self.headless_mode:
            document.upsert(
                PromptBlock(
                    name="headless_mode",
                    content=self._resolve_optional_prompt_section(
                        "HEADLESS_MODE_GUIDANCE",
                        HEADLESS_MODE_GUIDANCE,
                    ),
                    priority=25,
                    header="",
                    kind="guidance",
                )
            )
            logger.debug("Headless mode enabled - added automated execution guidance to prompt")

        if "contextual_guidance" in sections_to_include and self.contextual_guidance:
            document.upsert(
                PromptBlock(
                    name="contextual_guidance",
                    content=self.contextual_guidance,
                    priority=28,
                    header="### Contextual Constraints & Guidance",
                    kind="guidance",
                )
            )

        if "mode_guidance" in sections_to_include:
            mode_guidance = self._get_mode_guidance_section()
            if mode_guidance:
                document.upsert(
                    PromptBlock(
                        name="mode_guidance",
                        content=mode_guidance,
                        priority=30,
                        header="",
                        kind="guidance",
                    )
                )

        if "task_guidance" in sections_to_include:
            task_guidance = self._get_task_guidance_section()
            if task_guidance:
                document.upsert(
                    PromptBlock(
                        name="task_guidance",
                        content=task_guidance,
                        priority=40,
                        header="",
                        kind="guidance",
                    )
                )

        tool_constraint = ""
        if "tool_constraint" in sections_to_include:
            tool_constraint = self._get_tool_constraint_section()
            if tool_constraint:
                document.upsert(
                    PromptBlock(
                        name="tool_constraint",
                        content=tool_constraint,
                        priority=50,
                        header="",
                        kind="tooling",
                    )
                )

        if "completion" in sections_to_include:
            document.upsert(
                PromptBlock(
                    name="completion_guidance",
                    content=self._resolve_optional_prompt_section(
                        "COMPLETION_GUIDANCE",
                        COMPLETION_GUIDANCE,
                    ),
                    priority=60,
                    header="",
                    kind="completion",
                )
            )

        provider_tool_guidance = ""
        if "tool_guidance" in sections_to_include:
            provider_tool_guidance = self.get_provider_tool_guidance()
            if provider_tool_guidance:
                document.upsert(
                    PromptBlock(
                        name="provider_tool_guidance",
                        content=provider_tool_guidance,
                        priority=70,
                        header="",
                        kind="tooling",
                    )
                )

            document.upsert(
                PromptBlock(
                    name="tool_effectiveness_guidance",
                    content=self._resolve_optional_prompt_section(
                        "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
                        ASI_TOOL_EFFECTIVENESS_GUIDANCE,
                    ),
                    priority=80,
                    header="",
                    kind="tooling",
                )
            )

        rendered = document.render()
        logger.debug(
            "[SystemPrompt→LLM] provider=%s sections=%s tool_constraint=%s "
            "tool_guidance_len=%d prompt_total_len=%d strategy=%s",
            self.provider_name,
            sorted(sections_to_include),
            tool_constraint[:300] if tool_constraint else "(none)",
            len(provider_tool_guidance),
            len(rendered),
            self.system_prompt_strategy,
        )
        return document

    def invalidate_cache(self) -> None:
        """Clear any cached prompt so runtime state changes are reflected."""
        self._cached_prompt = None

    def _get_effective_strategy(self) -> str:
        """Get the effective strategy based on configuration and provider type.

        Returns:
            'static', 'dynamic', or 'hybrid'
        """
        strategy = self.system_prompt_strategy

        if strategy == "hybrid":
            # Static for API providers (cache benefit), dynamic for local
            if self.provider_caches or self.provider_has_kv_cache:
                return "static"
            else:
                return "dynamic"

        return strategy

    def _build_prompt_internal(self) -> str:
        """Internal method to build the prompt without caching logic.

        Returns:
            System prompt string tailored to the provider/model
        """
        return self.build_document().render()

    def _get_active_sections(self) -> set:
        """Determine which prompt sections to include.

        Uses edge model to select relevant sections when available,
        reducing system prompt token count for focused tasks.
        Falls back to all sections if edge model unavailable.
        """
        optional_sections = {
            "concise_mode",
            "headless_mode",
            "completion",
            "tool_guidance",
        }
        baseline_sections = {
            "mode_guidance",
            "task_guidance",
            "tool_constraint",
            "contextual_guidance",
        }
        all_sections = baseline_sections | optional_sections

        service = getattr(self, "_llm_decision_service", None)
        if service is None:
            service = self._resolve_llm_decision_service()
            self._llm_decision_service = service
        if service is not None:
            try:
                from victor.agent.prompt_section_registry import get_edge_focus_sections
                from victor.agent.edge_model import (
                    select_prompt_sections_with_edge_model,
                )

                # Use cached task type from classification if available
                task_type = getattr(self, "_task_type", "action")
                user_msg = getattr(self, "_user_message", "")
                available_sections = [section.name for section in get_edge_focus_sections()]

                selected = select_prompt_sections_with_edge_model(
                    service=service,
                    user_message=user_msg[:200] if user_msg else "",
                    task_type=task_type,
                    available_sections=available_sections,
                )

                if selected:
                    result = baseline_sections | self._map_edge_focus_to_builder_sections(
                        set(selected)
                    )
                    # Always include completion guidance (required for detection)
                    result.add("completion")
                    logger.debug(f"Edge prompt focus: {len(result)}/{len(all_sections)} sections")
                    return result

            except Exception:
                pass

        # For non-caching providers without edge model, use a reduced set.
        # Full sections are expensive (reparsed every turn) with no cache benefit.
        if not self.provider_caches:
            reduced = {
                "completion",
                "mode_guidance",
                "task_guidance",
                "tool_constraint",
            }
            if self.concise_mode:
                reduced.add("concise_mode")
            if self.headless_mode:
                reduced.add("headless_mode")
            logger.debug(
                f"Non-caching provider: using reduced sections {len(reduced)}/{len(all_sections)}"
            )
            return reduced

        return all_sections

    def _map_edge_focus_to_builder_sections(self, selected: set[str]) -> set[str]:
        """Map shared edge-focus selectors onto builder-local section keys."""
        mapped: set[str] = set()
        if "completion" in selected:
            mapped.add("completion")
        if "tool_guidance" in selected:
            mapped.add("tool_guidance")
        if "concise_mode" in selected and self.concise_mode:
            mapped.add("concise_mode")
        return mapped

    def _build_with_adapter(self) -> str:
        """Build system prompt using the tool calling adapter.

        Returns:
            System prompt string tailored to the provider/model
        """
        base_prompt = (
            "You are an expert coding assistant. You can analyze, explain, and generate code.\n"
            "When asked to write or complete code, provide working implementations directly.\n"
            "When asked to explore or analyze code, use the available tools."
        )

        # Get adapter-specific hints
        hints = self.tool_adapter.get_system_prompt_hints() if self.tool_adapter else None

        if hints:
            return f"{base_prompt}\n\n{hints}\n\n{GROUNDING_RULES}"

        # For providers with robust native tool calling, use minimal prompt
        caps = self.capabilities or (
            self.tool_adapter.get_capabilities() if self.tool_adapter else None
        )
        if caps and caps.native_tool_calls and not caps.requires_strict_prompting:
            return (
                f"{base_prompt}\n\n"
                "Tool usage guidelines:\n"
                "1. For code generation tasks: write the code directly in your response.\n"
                "2. For exploration tasks: use ls and read to examine code.\n"
                "3. For modification tasks: use write or edit "
                "after understanding context.\n"
                "4. Provide clear, working solutions.\n\n"
                f"{GROUNDING_RULES}"
            )

        return f"{base_prompt}\n\n{GROUNDING_RULES}"

    def _build_for_provider(self) -> str:
        """Build an appropriate system prompt based on the provider type.

        Returns:
            Appropriate system prompt string for the provider
        """
        # Cloud providers with robust native tool calling
        if self.is_cloud_provider():
            return self._build_cloud_prompt()

        # vLLM - Production-grade OpenAI-compatible
        if self.provider_name == "vllm":
            return self._build_vllm_prompt()

        # LMStudio - OpenAI-compatible with Native vs Default mode
        if self.provider_name == "lmstudio":
            return self._build_lmstudio_prompt()

        # Ollama - Native tool_calls for supported models
        if self.provider_name == "ollama":
            return self._build_ollama_prompt()

        # Default/unknown provider
        return self._build_default_prompt()

    def _build_cloud_prompt(self) -> str:
        """Build prompt for cloud providers (Anthropic, OpenAI, xAI, DeepSeek).

        Delegates to provider-specific prompts for optimal behavior.
        """
        if self.provider_name == "google":
            return self._build_google_prompt()

        if self.provider_name == "deepseek":
            return self._build_deepseek_prompt()

        if self.provider_name == "xai":
            return self._build_xai_prompt()

        return (
            "You are an expert code analyst with access to tools for exploring "
            "and modifying code. Use them effectively:\n\n"
            "1. Use ls and read to examine code before conclusions.\n"
            "2. If asked to modify code, use write or edit "
            "after understanding context.\n"
            "3. For call-graph questions, prefer graph(mode='callers'|'callees'|'trace').\n"
            "4. Provide clear, actionable responses based on actual file contents.\n"
            "5. Always cite specific file paths and line numbers when referencing code.\n"
            "6. You may call multiple tools in parallel when they are independent.\n\n"
            f"{self._resolve_optional_prompt_section('PARALLEL_READ_GUIDANCE', PARALLEL_READ_GUIDANCE)}\n\n"
            f"{GROUNDING_RULES}"
        )

    def _build_google_prompt(self) -> str:
        """Build optimized prompt for Google/Gemini models.

        Gemini Flash/Pro have excellent native tool calling and work best with:
        - Concise, action-oriented instructions
        - Minimal redundancy (don't repeat what's in tool definitions)
        - Focus on task completion over verbose guidance
        """
        return (
            "Expert coding assistant with tool access.\n\n"
            "CORE RULES:\n"
            "• Read files before making claims about their contents\n"
            "• Cite file:line when referencing code\n"
            "• Parallel tool calls allowed for independent operations\n"
            "• Ground all responses in actual tool output\n\n"
            "TASK EXECUTION:\n"
            "• Generation: Write code directly, minimal tool use\n"
            "• Exploration: Use tools systematically, then summarize\n"
            "• Modification: Read → understand → edit\n"
            "• For call-graph questions, use graph(mode='callers'|'callees'|'trace')\n"
            "• Actions: Execute fully, report results\n\n"
            f"{self._resolve_optional_prompt_section('PARALLEL_READ_GUIDANCE', PARALLEL_READ_GUIDANCE)}\n\n"
            f"{GROUNDING_RULES}"
        )

    def _build_deepseek_prompt(self) -> str:
        """Build optimized prompt for DeepSeek models.

        DeepSeek models have good tool calling but tend to:
        - Read the same file multiple times (repetitive tool calls)
        - Generate code snippets without verification
        - Need explicit grounding reminders

        This prompt emphasizes:
        - Anti-repetition rules
        - Strict grounding requirements
        - Efficient tool usage patterns
        - Pagination for large files
        """
        return (
            "Expert coding assistant with tool access.\n\n"
            "CRITICAL RULES (MUST FOLLOW):\n"
            "• NEVER read the same file twice with identical arguments\n"
            "• NEVER call the same tool with identical arguments\n"
            "• If you've read a file, use that content for all future references\n"
            "• Only call tools when you need NEW information\n\n"
            f"{self._resolve_optional_prompt_section('LARGE_FILE_PAGINATION_GUIDANCE', LARGE_FILE_PAGINATION_GUIDANCE)}\n\n"
            "GROUNDING (MANDATORY):\n"
            "• ALL code snippets must be directly from tool output\n"
            "• Do NOT generate or imagine file contents\n"
            "• Quote code EXACTLY as it appears in tool output\n"
            "• If unsure about content, read the file ONCE\n"
            "• Never cite line numbers you haven't verified\n\n"
            "TOOL EFFICIENCY:\n"
            "• ls first to understand structure\n"
            "• read ONCE per file, remember contents\n"
            "• For large files, use search/offset parameters (see above)\n"
            "• Use semantic_code_search for specific symbols\n"
            "• Use graph(mode='callers'|'callees'|'trace') for call-graph questions\n"
            "• Stop tool calls when you have enough info (usually 3-5 calls)\n\n"
            "TASK EXECUTION:\n"
            "• Generation: Write code directly without excessive exploration\n"
            "• Analysis: Read files ONCE, provide grounded analysis\n"
            "• Modification: Read target file ONCE, then edit\n\n"
            f"{self._resolve_optional_prompt_section('GROUNDING_RULES_EXTENDED', GROUNDING_RULES_EXTENDED)}"
        )

    def _get_evolved_content_resolver(self) -> Any:
        """Lazily create the scoped evolved-content resolver."""
        if self._evolved_content_resolver is not None:
            return self._evolved_content_resolver

        try:
            from victor.agent.evolved_content_resolver import EvolvedContentResolver
            from victor.agent.optimization_injector import OptimizationInjector

            self._evolved_content_resolver = EvolvedContentResolver(
                optimization_injector=OptimizationInjector()
            )
        except Exception:
            self._evolved_content_resolver = False
        return self._evolved_content_resolver

    def _resolve_optional_prompt_section(self, section_name: str, fallback_text: str) -> str:
        """Resolve scoped evolvable prompt text with safe fallback."""
        resolver = self._get_evolved_content_resolver()
        if resolver in (None, False):
            return fallback_text

        try:
            resolved = resolver.resolve_section(
                section_name=section_name,
                provider=self.provider_name,
                model=self.model,
                task_type=self.task_type,
                fallback_text=fallback_text,
            )
            return resolved.text or fallback_text
        except Exception:
            return fallback_text

    def _build_xai_prompt(self) -> str:
        """Build optimized prompt for xAI/Grok models.

        Grok models are generally good at tool calling but benefit from:
        - Clear task structure
        - Explicit completion criteria
        - Balanced exploration guidance
        """
        return (
            "You are an expert code analyst with access to tools.\n\n"
            "EFFECTIVE TOOL USAGE:\n"
            "• Use ls to understand project structure first\n"
            "• Use read to examine specific files (one read per file)\n"
            "• Use semantic_code_search for finding specific code patterns\n"
            "• Use graph(mode='callers'|'callees'|'trace') for call-graph questions\n"
            "• Parallel tool calls are allowed for independent operations\n\n"
            "TASK APPROACH:\n"
            "• For analysis tasks: Read relevant files, then provide structured findings\n"
            "• For generation tasks: Write code directly with minimal exploration\n"
            "• For modification tasks: Read → understand → modify\n"
            "• Stop exploring when you have sufficient information\n\n"
            "RESPONSE QUALITY:\n"
            "• Cite specific file paths and line numbers\n"
            "• Base all claims on actual tool output\n"
            "• Provide actionable, concrete suggestions\n"
            "• Structure responses clearly with headers when appropriate\n\n"
            f"{self._resolve_optional_prompt_section('PARALLEL_READ_GUIDANCE', PARALLEL_READ_GUIDANCE)}\n\n"
            f"{GROUNDING_RULES}"
        )

    def _get_provider_tool_hint_block(self, provider_name: Optional[str] = None) -> str:
        """Resolve provider-specific adapter hints for local provider prompt builders."""
        provider_key = (provider_name or self.provider_name or "").lower()
        if not provider_key:
            return ""

        adapter = None
        if self.tool_adapter is not None:
            adapter_provider = str(getattr(self.tool_adapter, "provider_name", "") or "").lower()
            if adapter_provider == provider_key:
                adapter = self.tool_adapter

        if adapter is None:
            try:
                from victor.agent.tool_calling.adapters import (
                    LMStudioToolCallingAdapter,
                    OllamaToolCallingAdapter,
                    OpenAICompatToolCallingAdapter,
                )

                adapter_cls = {
                    "ollama": OllamaToolCallingAdapter,
                    "lmstudio": LMStudioToolCallingAdapter,
                    "vllm": OpenAICompatToolCallingAdapter,
                }.get(provider_key)
                if adapter_cls is None:
                    return ""
                adapter = adapter_cls(self.model, None)
            except Exception:
                logger.debug(
                    "Failed to initialize provider hint adapter for %s",
                    provider_key,
                    exc_info=True,
                )
                return ""

        try:
            if provider_key in {"ollama", "lmstudio"} and hasattr(adapter, "get_capabilities"):
                capabilities = adapter.get_capabilities()
                expected_native = self.has_native_tool_support()
                expected_thinking = "qwen3" in self.model_lower or "qwen-3" in self.model_lower
                if (
                    capabilities.native_tool_calls != expected_native
                    or capabilities.thinking_mode != expected_thinking
                ):
                    adjusted_capabilities = ToolCallingCapabilities(
                        native_tool_calls=expected_native,
                        streaming_tool_calls=capabilities.streaming_tool_calls,
                        parallel_tool_calls=capabilities.parallel_tool_calls,
                        tool_choice_param=capabilities.tool_choice_param,
                        json_fallback_parsing=capabilities.json_fallback_parsing,
                        xml_fallback_parsing=capabilities.xml_fallback_parsing,
                        thinking_mode=expected_thinking,
                        requires_strict_prompting=capabilities.requires_strict_prompting,
                        tool_call_format=capabilities.tool_call_format,
                        argument_format=capabilities.argument_format,
                        recommended_max_tools=capabilities.recommended_max_tools,
                        recommended_tool_budget=capabilities.recommended_tool_budget,
                    )
                    adapter.get_capabilities = lambda: adjusted_capabilities
            return str(adapter.get_system_prompt_hints() or "").strip()
        except Exception:
            logger.debug(
                "Failed to resolve provider tool hints for %s",
                provider_key,
                exc_info=True,
            )
            return ""

    def _build_vllm_prompt(self) -> str:
        """Build prompt for vLLM provider."""
        adapter_hints = self._get_provider_tool_hint_block("vllm")
        if adapter_hints:
            return (
                "You are a code analyst. You have access to tools via OpenAI-compatible API.\n\n"
                f"{adapter_hints}\n\n"
                "IMPORTANT:\n"
                "- Do not output raw JSON tool calls in your text response.\n"
                "- Do not output XML tags like </function> or </parameter>.\n"
                "- When you're done using tools, provide a human-readable answer.\n\n"
                f"{GROUNDING_RULES}"
            )
        return (
            "You are a code analyst. You have access to tools via OpenAI-compatible API.\n\n"
            "TOOL CALLING:\n"
            "- Tools are called using the standard OpenAI function calling format.\n"
            "- You may call tools when needed to gather information.\n"
            "- After gathering sufficient information (2-4 tool calls), provide your answer.\n"
            "- Do NOT repeat the same tool call with identical arguments.\n\n"
            "RESPONSE FORMAT:\n"
            "- When you have enough information, respond with a clear answer.\n"
            "- Your final response should be in plain text, not JSON or tool syntax.\n"
            "- Be concise and focus on answering the user's question.\n\n"
            "IMPORTANT:\n"
            "- Do not output raw JSON tool calls in your text response.\n"
            "- Do not output XML tags like </function> or </parameter>.\n"
            "- When you're done using tools, provide a human-readable answer.\n\n"
            f"{GROUNDING_RULES}"
        )

    def _build_lmstudio_prompt(self) -> str:
        """Build prompt for LMStudio provider."""
        adapter_hints = self._get_provider_tool_hint_block("lmstudio")
        if self.has_native_tool_support():
            if adapter_hints:
                return (
                    "You are an expert coding assistant. "
                    "You can analyze, explain, and generate code.\n\n"
                    "CAPABILITIES:\n"
                    "- Code generation: Write working implementations directly in your response.\n"
                    "- Code analysis: Use tools to explore and understand existing code.\n"
                    "- Code modification: Use tools to read files before making changes.\n\n"
                    f"{adapter_hints}\n\n"
                    "RESPONSE FORMAT:\n"
                    "- For code generation, include the complete implementation.\n"
                    "- Do NOT include JSON, XML, or tool syntax in your response.\n\n"
                    f"{GROUNDING_RULES}"
                )
            return (
                "You are an expert coding assistant. "
                "You can analyze, explain, and generate code.\n\n"
                "CAPABILITIES:\n"
                "- Code generation: Write working implementations directly in your response.\n"
                "- Code analysis: Use tools to explore and understand existing code.\n"
                "- Code modification: Use tools to read files before making changes.\n\n"
                "TOOL USAGE:\n"
                "- For code generation tasks: write code directly, no tools needed.\n"
                "- For exploration tasks: use tools via OpenAI function calling format.\n"
                "- Call tools one at a time and wait for results.\n"
                "- After 2-3 tool calls, provide your answer.\n"
                "- Do NOT repeat identical tool calls.\n\n"
                "RESPONSE FORMAT:\n"
                "- Provide answers in plain, readable text.\n"
                "- For code generation, include the complete implementation.\n"
                "- Do NOT include JSON, XML, or tool syntax in your response.\n\n"
                f"{GROUNDING_RULES}"
            )
        else:
            # Default/non-native mode - stricter guidance needed
            if adapter_hints:
                return (
                    "You are a code analyst. Follow these rules EXACTLY:\n\n"
                    f"{adapter_hints}\n\n"
                    "WHEN TO STOP:\n"
                    "1. When you have read the relevant files.\n"
                    "2. When you can answer the user's question.\n"
                    "3. After calling any tool 3 times.\n\n"
                    f"{GROUNDING_RULES}"
                )
            return (
                "You are a code analyst. Follow these rules EXACTLY:\n\n"
                "CRITICAL RULES:\n"
                "1. Call tools ONE AT A TIME. Wait for each result.\n"
                "2. After reading 2-3 files, STOP and provide your answer.\n"
                "3. Do NOT repeat the same tool call.\n\n"
                "OUTPUT FORMAT:\n"
                "1. Your answer must be in plain English text.\n"
                "2. Do NOT output JSON objects in your response.\n"
                "3. Do NOT output XML tags like </function> or <parameter>.\n"
                "4. Do NOT output [TOOL_REQUEST] or similar markers.\n\n"
                "WHEN TO STOP:\n"
                "1. When you have read the relevant files.\n"
                "2. When you can answer the user's question.\n"
                "3. After calling any tool 3 times.\n\n"
                f"{GROUNDING_RULES}"
            )

    def _build_ollama_prompt(self) -> str:
        """Build prompt for Ollama provider."""
        adapter_hints = self._get_provider_tool_hint_block("ollama")
        if self.has_native_tool_support():
            if adapter_hints:
                return (
                    "You are a code analyst with tool calling capability.\n\n"
                    f"{adapter_hints}\n\n"
                    f"{self._resolve_optional_prompt_section('PARALLEL_READ_GUIDANCE', PARALLEL_READ_GUIDANCE)}\n\n"
                    "COMPLETION:\n"
                    "- Stop calling tools when you have enough information.\n"
                    "- If you've called a tool 3+ times, stop and summarize.\n"
                    "- Always end with a human-readable answer.\n\n"
                    f"{GROUNDING_RULES}"
                )
            base_prompt = (
                "You are a code analyst with tool calling capability.\n\n"
                "TOOL USAGE:\n"
                "- Use ls and read to inspect code.\n"
                "- You can call multiple read tools in parallel for efficiency.\n"
                "- After reading relevant files, provide your answer.\n"
                "- Do NOT make identical repeated tool calls.\n\n"
                f"{self._resolve_optional_prompt_section('PARALLEL_READ_GUIDANCE', PARALLEL_READ_GUIDANCE)}\n\n"
                "RESPONSE FORMAT:\n"
                "- Write your answer in plain, readable text.\n"
                "- Do NOT output raw JSON in your response.\n"
                "- Do NOT output XML tags or function call syntax.\n"
                "- Be concise and answer the question directly.\n\n"
                "COMPLETION:\n"
                "- Stop calling tools when you have enough information.\n"
                "- If you've called a tool 3+ times, stop and summarize.\n"
                "- Always end with a human-readable answer.\n\n"
                f"{GROUNDING_RULES}"
            )
            return base_prompt
        else:
            # Models without reliable tool calling - strictest guidance
            if adapter_hints:
                return (
                    "You are a code analyst. Follow these rules EXACTLY:\n\n"
                    f"{adapter_hints}\n\n"
                    "STOP IMMEDIATELY WHEN:\n"
                    "1. You have read the relevant files.\n"
                    "2. You can answer the user's question.\n"
                    "3. You have called any tool 3+ times.\n\n"
                    f"{GROUNDING_RULES}"
                )
            return (
                "You are a code analyst. Follow these rules EXACTLY:\n\n"
                "CRITICAL TOOL RULES:\n"
                "1. Call tools ONE AT A TIME. Never batch calls.\n"
                "2. After reading 2-3 files, STOP and answer.\n"
                "3. Do NOT repeat the same tool call.\n\n"
                "CRITICAL OUTPUT RULES:\n"
                "1. Write your answer in plain English.\n"
                '2. Do NOT output JSON objects like {"name": ...}.\n'
                "3. Do NOT output XML tags like </function> or </parameter>.\n"
                "4. Do NOT output function call syntax.\n"
                "5. Keep your answer focused and concise.\n\n"
                "STOP IMMEDIATELY WHEN:\n"
                "1. You have read the relevant files.\n"
                "2. You can answer the user's question.\n"
                "3. You have called any tool 3+ times.\n\n"
                f"{GROUNDING_RULES}"
            )

    def _build_default_prompt(self) -> str:
        """Build default prompt for unknown providers."""
        return (
            "You are a code analyst. Follow these rules strictly:\n\n"
            "TOOL USAGE:\n"
            "- Use ls or read to inspect files before answering.\n"
            "- Call tools ONE AT A TIME. Wait for results before calling the next tool.\n"
            "- After reading 2-3 relevant files, STOP and provide your answer.\n"
            "- Do NOT repeatedly call the same tool with similar arguments.\n\n"
            "RESPONSE FORMAT:\n"
            "- After gathering information, provide a CLEAR ANSWER in plain text.\n"
            "- Do NOT output raw JSON, XML tags, or tool call syntax in your response.\n"
            "- Keep responses concise and focused on the user's question.\n\n"
            "WHEN TO STOP:\n"
            "- Stop calling tools when you have enough information to answer.\n"
            "- If you've called the same tool 3+ times, stop and summarize.\n"
            "- Always end with a human-readable answer, not more tool calls.\n\n"
            f"{GROUNDING_RULES}"
        )


def build_system_prompt(
    provider_name: str,
    model: str,
    tool_adapter: Optional[BaseToolCallingAdapter] = None,
    capabilities: Optional[ToolCallingCapabilities] = None,
    prompt_contributors: Optional[list] = None,
    concise_mode: bool = False,
) -> str:
    """Build a system prompt (convenience function).

    Args:
        provider_name: Provider name
        model: Model name
        tool_adapter: Optional tool calling adapter
        capabilities: Optional pre-computed capabilities
        prompt_contributors: Optional list of PromptContributorProtocol implementations
        concise_mode: If True, adds guidance for brief, direct responses

    Returns:
        System prompt string
    """
    builder = SystemPromptBuilder(
        provider_name=provider_name,
        model=model,
        tool_adapter=tool_adapter,
        capabilities=capabilities,
        prompt_contributors=prompt_contributors,
        concise_mode=concise_mode,
    )
    return builder.build()
