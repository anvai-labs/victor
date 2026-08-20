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

"""CodingAssistant - Victor's primary vertical for software development.

This module defines the CodingAssistant vertical with full integration
of coding-specific extensions, middleware, and configurations.

The CodingAssistant provides:
- 45+ tools optimized for coding tasks
- Stage-aware tool selection for workflow optimization
- Code validation and correction middleware
- Git operation safety checks
- Task-type-specific prompt hints
- Mode configurations for different coding scenarios
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from victor_contracts import (
    FileOperationsCapability,
    PromptContributionCapability,
    StageDefinition,
    ToolNames,
    VerticalBase,
    VerticalConfig,
)
from victor_contracts.verticals import (
    MiddlewareProtocol,
    ServiceProviderProtocol,
    register_vertical,
)


@register_vertical(
    name="coding",
    version="2.0.0",
    min_framework_version=">=0.6.0",
    canonicalize_tool_names=True,
    tool_dependency_strategy="auto",
    strict_mode=False,
    load_priority=100,  # High priority - default vertical
    plugin_namespace="victor.coding",
)
class CodingAssistant(VerticalBase):
    """Software development assistant vertical.

    This is Victor's default configuration, optimized for:
    - Code exploration and understanding
    - Bug fixing and refactoring
    - Feature implementation
    - Testing and verification
    - Git operations and version control

    The CodingAssistant provides full integration with the framework
    through extension protocols, enabling:
    - Code correction middleware for validation
    - Git safety checks for dangerous operations
    - Task-type-specific prompt hints
    - Mode configurations for different scenarios
    - Tool dependency graph for intelligent selection

    Example:
        from victor_coding import CodingAssistant

        # Get vertical configuration
        config = CodingAssistant.get_config()

        # Get extensions for framework integration
        extensions = CodingAssistant.get_extensions()

        # Create agent with this vertical
        agent = await Agent.create(
            tools=config.tools,
            vertical=CodingAssistant,
        )
    """

    name = "coding"
    description = "Software development assistant for code exploration, writing, and refactoring"
    version = "2.0.0"  # Extension support
    VERTICAL_API_VERSION = 1

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_skills(cls) -> list:
        """Return coding skills for this vertical."""
        from victor_coding.skills import CODING_SKILLS

        return list(CODING_SKILLS)

    @classmethod
    def get_lsp(cls):
        """Expose victor-coding's LSP capability to the framework (FEP-0019).

        Returns a ``FrameworkLSPAdapter`` over the shared ``LSPConnectionPool``.
        The framework duck-types ``orchestrator.lsp``, and
        ``FrameworkStepHandler.apply_lsp`` routes this object to
        ``orchestrator.set_lsp`` during integration, activating the diagnostic
        middleware (Phases 1+2) and the symbol-context provider (Phase 3).
        Returns ``None`` if the pool/adapter cannot be built.
        """
        try:
            from victor_coding.lsp.framework_adapter import FrameworkLSPAdapter
            from victor_coding.lsp.manager import get_lsp_manager

            return FrameworkLSPAdapter(get_lsp_manager())
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "victor-coding LSP capability unavailable; FEP-0019 chain stays dormant",
                exc_info=True,
            )
            return None

    # =========================================================================
    # Phase 3: Framework Capabilities
    # =========================================================================
    # Reuse framework capabilities to reduce code duplication

    # Framework file operations capability (read, write, edit, grep)
    _file_ops = FileOperationsCapability()

    # Framework prompt contributions (common hints like read_first, verify_changes)
    _prompt_contrib = PromptContributionCapability()

    # =========================================================================
    # Extension Caching
    # =========================================================================
    # Individual extension caching is provided by VerticalBase._get_cached_extension()
    # Composite extensions caching is provided by VerticalBase.get_extensions()
    # Use clear_config_cache() to invalidate all caches.

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for software development.

        Phase 3: Uses framework FileOperationsCapability for common file operations
        to reduce code duplication and maintain consistency across verticals.

        Uses canonical tool names from victor.tools.tool_names.

        Returns:
            List of tool names including filesystem, git, shell, and code tools.
        """
        # Phase 3: Start with framework file operations (read, write, edit, grep)
        # This reduces duplication and ensures consistency across verticals
        tools = cls._file_ops.get_tool_list()

        # Add coding-specific tools
        tools.extend(
            [
                # Core filesystem (beyond framework basics)
                ToolNames.LS,  # list_directory -> ls
                ToolNames.OVERVIEW,  # get_project_overview -> overview
                # Search
                ToolNames.CODE_SEARCH,  # semantic_code_search -> code_search
                ToolNames.PLAN,  # plan_files -> plan
                # Git (unified git tool handles all operations)
                ToolNames.GIT,  # Git operations
                # Shell
                ToolNames.SHELL,  # execute_bash -> shell
                # Code intelligence
                ToolNames.LSP,  # lsp operations
                ToolNames.SYMBOL,  # find_symbol -> symbol
                ToolNames.REFS,  # find_references -> refs
                # Refactoring
                ToolNames.RENAME,  # refactor_rename_symbol -> rename
                ToolNames.EXTRACT,  # refactor_extract_function -> extract
                # Testing
                ToolNames.TEST,  # run_tests -> test
                # Docker
                ToolNames.DOCKER,  # docker operations
                # Web (for documentation)
                ToolNames.WEB_SEARCH,  # web_search
                ToolNames.WEB_FETCH,  # web_fetch
            ]
        )

        return tools

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get coding-focused system prompt.

        Returns:
            System prompt optimized for software development.
        """
        return """You are Victor, an expert software development assistant.

Your capabilities:
- Deep code understanding through semantic search and LSP integration
- Safe file operations with automatic backup and undo
- Git operations for version control
- Test execution and validation
- Multi-language support (Python, TypeScript, Rust, Go, and more)

Guidelines:
1. **Understand before modifying**: Always read and understand code before making changes
2. **Incremental changes**: Make small, focused changes rather than large rewrites
3. **Verify changes**: Run tests or validation after modifications
4. **Explain reasoning**: Briefly explain your approach when making non-trivial changes
5. **Preserve style**: Match existing code style and patterns
6. **Handle errors gracefully**: If something fails, diagnose and recover

When exploring code:
- Use semantic_code_search for conceptual queries ("authentication logic")
- Use code_search for exact patterns ("def authenticate")
- Use overview to understand file structure

When modifying code:
- Use edit for surgical changes to existing code
- Use write only for new files or complete rewrites
- Always verify changes compile/pass tests when possible

You have access to 45+ tools. Use them efficiently to accomplish tasks."""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get coding-specific stage definitions.

        Uses canonical tool names from victor.tools.tool_names.

        Returns:
            Stage definitions optimized for software development workflow.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the coding request",
                tools={
                    ToolNames.READ,
                    ToolNames.LS,
                    ToolNames.OVERVIEW,
                    ToolNames.GREP,
                },
                keywords=["what", "how", "explain", "where", "show me"],
                next_stages={"PLANNING", "READING"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning the implementation approach",
                tools={
                    ToolNames.GREP,
                    ToolNames.PLAN,
                    ToolNames.OVERVIEW,
                    ToolNames.READ,
                },
                keywords=["plan", "approach", "design", "architecture", "strategy"],
                next_stages={"READING", "EXECUTION"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Reading code and gathering context",
                tools={
                    ToolNames.READ,
                    ToolNames.CODE_SEARCH,
                    ToolNames.GREP,
                    ToolNames.LSP,
                    ToolNames.SYMBOL,
                    ToolNames.REFS,
                },
                keywords=["read", "show", "find", "look", "check", "search"],
                next_stages={"ANALYSIS", "EXECUTION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Analyzing code structure and dependencies",
                tools={
                    ToolNames.LSP,
                    ToolNames.SYMBOL,
                    ToolNames.REFS,
                    ToolNames.OVERVIEW,
                },
                keywords=["analyze", "review", "understand", "why", "how does"],
                next_stages={"EXECUTION", "PLANNING"},
            ),
            "EXECUTION": StageDefinition(
                name="EXECUTION",
                description="Implementing changes",
                tools={
                    ToolNames.WRITE,
                    ToolNames.EDIT,
                    ToolNames.SHELL,
                    ToolNames.GIT,
                    ToolNames.RENAME,
                },
                keywords=[
                    "change",
                    "modify",
                    "create",
                    "add",
                    "remove",
                    "fix",
                    "implement",
                    "write",
                    "update",
                    "refactor",
                ],
                next_stages={"VERIFICATION", "COMPLETION"},
            ),
            "VERIFICATION": StageDefinition(
                name="VERIFICATION",
                description="Testing and validating changes",
                tools={ToolNames.SHELL, ToolNames.TEST, ToolNames.GIT, ToolNames.READ},
                keywords=["test", "verify", "check", "validate", "run", "build"],
                next_stages={"COMPLETION", "EXECUTION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Committing and summarizing",
                tools={ToolNames.GIT},
                keywords=["done", "finish", "complete", "commit", "summarize"],
                next_stages=set(),
            ),
        }

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        """Add coding-specific configuration.

        Args:
            config: Base configuration.

        Returns:
            Customized configuration.
        """
        config.metadata["supports_lsp"] = True
        config.metadata["supports_git"] = True
        config.metadata["max_file_size"] = 1_000_000  # 1MB
        config.metadata["supported_languages"] = [
            "python",
            "typescript",
            "javascript",
            "rust",
            "go",
            "java",
            "c",
            "cpp",
        ]
        return config

    # =========================================================================
    # Extension Protocol Methods
    # =========================================================================

    @classmethod
    def get_middleware(cls) -> List[MiddlewareProtocol]:
        """Get coding-specific middleware (cached).

        Returns:
            List of middleware implementations
        """

        def _create_middleware() -> List[MiddlewareProtocol]:
            from victor_coding.middleware import (
                CodeCorrectionMiddleware,
                GitSafetyMiddleware,
            )

            return [
                CodeCorrectionMiddleware(enabled=True, auto_fix=True),
                GitSafetyMiddleware(block_dangerous=False, warn_on_risky=True),
            ]

        return cls._get_cached_extension("middleware", _create_middleware)

    @classmethod
    def get_service_provider(cls) -> Optional[ServiceProviderProtocol]:
        """Get coding-specific service provider (cached).

        Returns:
            Service provider for DI registration
        """
        return cls._get_extension_factory("service_provider", "victor_coding.service_provider")

    @classmethod
    def get_composed_chains(cls) -> Dict[str, Any]:
        """Get pre-built LCEL-composed tool chains (cached).

        Provides LCEL composition chains for common coding tasks:
        - explore_file: Read file and analyze symbols
        - analyze_function: Get function details with references
        - safe_edit: Edit with verification
        - git_status: Parallel git state collection
        - search_with_context: Code search with result context
        - lint: Language-aware linting
        - test_discovery: Find test files
        - review_analysis: Parallel review data collection

        Returns:
            Dict mapping chain names to Runnable instances
        """

        def _create() -> Dict[str, Any]:
            from victor_coding.composed_chains import CODING_CHAINS

            return CODING_CHAINS

        return cls._get_cached_extension("composed_chains", _create)

    @classmethod
    def get_personas(cls) -> Dict[str, Any]:
        """Get persona definitions for team members (cached).

        Provides rich persona definitions with:
        - Expertise categories
        - Communication styles
        - Decision-making preferences
        - Behavioral traits

        Available personas:
        - code_archaeologist: Deep code analysis expert
        - security_auditor: Security-focused reviewer
        - architect: Solution designer
        - refactoring_strategist: Safe refactoring planner
        - craftsman: Clean code implementer
        - debugger: Bug hunting specialist
        - quality_guardian: Code review expert
        - test_specialist: Testing expert

        Returns:
            Dict mapping persona names to CodingPersona instances
        """

        def _create() -> Dict[str, Any]:
            from victor_coding.teams import CODING_PERSONAS

            return CODING_PERSONAS

        return cls._get_cached_extension("personas", _create)

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        """Get coding capability configurations for centralized storage.

        Returns coding capability configurations for VerticalContext storage.
        This replaces direct orchestrator attribute assignment patterns like:
        - orchestrator.code_style = {...}
        - orchestrator.test_config = {...}
        - orchestrator.lsp_config = {...}

        Returns:
            Dict with coding capability configurations
        """
        from victor_coding.capabilities import get_capability_configs

        return get_capability_configs()

    @classmethod
    def get_capability_registrations(cls):
        """Return (protocol_type, provider) pairs for CapabilityRegistry.

        These capabilities are auto-registered when the plugin calls
        context.register_vertical(CodingAssistant). Lazy factories
        defer heavy imports until first use.
        """
        registrations = []
        try:
            from victor_contracts.capability_runtime import (
                CodebaseIndexFactoryProtocol,
                EditorProtocol,
                TreeSitterParserProtocol,
                create_lazy_capability_proxy,
            )

            # Tree-sitter parser (lazy — imports tree_sitter_manager on first access)
            def _load_ts():
                import importlib

                mod = importlib.import_module("victor_coding.codebase.tree_sitter_manager")
                return mod

            registrations.append((TreeSitterParserProtocol, create_lazy_capability_proxy(_load_ts)))

            # Editor (lazy)
            def _load_editor():
                import importlib

                mod = importlib.import_module("victor_coding.editing.editor")
                return mod.FileEditor()

            registrations.append((EditorProtocol, create_lazy_capability_proxy(_load_editor)))

            # Codebase index factory: register the REAL victor-coding CodebaseIndex
            # factory. Do NOT use detect_enhanced_index_factory() — it returns the
            # framework's delegating shell (EnhancedCodebaseIndexFactory), and
            # registering that shell as the provider creates a self-reference loop
            # (its create() finds only itself in the registry -> ImportError).
            def _load_index_factory():
                import inspect

                from victor_coding.codebase.indexer import CodebaseIndex

                # Forward only kwargs the installed CodebaseIndex accepts, so
                # callers targeting a newer/different signature (e.g. the
                # deprecated graph_writer_mode compat flag) don't break against
                # this version.
                _accepted = set(inspect.signature(CodebaseIndex.__init__).parameters) - {"self"}

                class _CodebaseIndexFactory:
                    def create(self, root_path, **kwargs):
                        filtered = {k: v for k, v in kwargs.items() if k in _accepted}
                        return CodebaseIndex(root_path, **filtered)

                return _CodebaseIndexFactory()

            registrations.append((CodebaseIndexFactoryProtocol, _load_index_factory()))

        except ImportError:
            pass  # Framework protocols not available (contract-only mode)

        return registrations

    # NOTE: get_extensions() is inherited from VerticalBase with full caching support.
    # Individual extension getters use _get_cached_extension() from VerticalBase.
    # To clear all caches, use cls.clear_config_cache().


__all__ = ["CodingAssistant"]
