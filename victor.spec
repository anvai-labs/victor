# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['victor/ui/cli.py'],
    pathex=[],
    binaries=[],
    datas=[('victor/config/tool_tiers.yaml', 'victor/config'), ('victor/config/logging_config.yaml', 'victor/config'), ('victor/config/provider_context_limits.yaml', 'victor/config'), ('victor/config/openai_compat_model_policy.yaml', 'victor/config'), ('victor/config/api_keys_registry.yaml', 'victor/config'), ('victor/config/tool_calling_models.yaml', 'victor/config'), ('victor/config/stage_keywords.yaml', 'victor/config'), ('victor/config/tool_categories.yaml', 'victor/config'), ('victor/config/rbac.yaml', 'victor/config'), ('victor/config/provider_metrics.yaml', 'victor/config'), ('victor/config/mode_coordination.yaml', 'victor/config'), ('victor/config/vertical_tools.yaml', 'victor/config'), ('victor/config/model_capabilities.yaml', 'victor/config'), ('victor/config/task_tool_config.yaml', 'victor/config'), ('victor/config/profiles.default.yaml', 'victor/config'), ('examples/profiles.yaml.example', 'examples')],
    hiddenimports=['victor', 'victor.ui', 'victor.ui.cli', 'victor.ui.commands', 'victor.agent', 'victor.agent.orchestrator', 'victor.agent.tool_executor', 'victor.agent.tool_calling', 'victor.agent.tool_calling.adapters', 'victor.agent.model_switcher', 'victor.agent.change_tracker', 'victor.agent.conversation_embedding_store', 'victor.providers', 'victor.providers.anthropic_provider', 'victor.providers.openai_provider', 'victor.providers.ollama_provider', 'victor.providers.google_provider', 'victor.providers.xai_provider', 'victor.tools', 'victor.tools.filesystem', 'victor.tools.code_search_tool', 'victor.tools.semantic_selector', 'victor.integrations.mcp', 'victor.integrations.mcp.server', 'victor.config', 'victor.config.settings', 'victor.evaluation', 'victor.evaluation.benchmarks', 'victor.evaluation.protocol', 'aiohttp', 'aiohttp.web', 'httpx', 'pydantic', 'pydantic_settings', 'yaml', 'tiktoken', 'tiktoken_ext', 'tiktoken_ext.openai_public', 'asyncio', 'aiofiles', 'jsonschema', 'pygments', 'victor.agent.protocols.agent_factory', 'victor.agent.protocols.analysis_protocols', 'victor.agent.protocols.budget_protocols', 'victor.agent.protocols.context_protocols', 'victor.agent.protocols.conversation_protocols', 'victor.agent.protocols.coordination_protocols', 'victor.agent.protocols.infrastructure_protocols', 'victor.agent.protocols.provider_protocols', 'victor.agent.protocols.streaming_protocols', 'victor.agent.protocols.task_completion', 'victor.agent.protocols.tool_protocols', 'victor.agent.coordinators.chat_protocols', 'victor.agent.coordinators.coordination_state_passed', 'victor.agent.coordinators.coordinator_factory', 'victor.agent.coordinators.exploration_state_passed', 'victor.agent.coordinators.factory_support', 'victor.agent.coordinators.planning_workflow', 'victor.agent.coordinators.protocol_dependencies', 'victor.agent.coordinators.protocols', 'victor.agent.coordinators.safety_state_passed', 'victor.agent.coordinators.stage_transition_coordinator', 'victor.agent.coordinators.state_context', 'victor.agent.coordinators.streaming_loop_handler', 'victor.agent.coordinators.system_prompt_state_passed', 'victor.agent.coordinators.transition_strategies', 'victor.agent.coordinators.turn_executor', 'victor.agent.coordinators.state_context', 'victor.agent.services.exploration_runtime', 'victor.agent.services.planning_runtime', 'victor.providers.anthropic_provider', 'victor.providers.azure_openai_provider', 'victor.providers.bedrock_provider', 'victor.providers.cerebras_provider', 'victor.providers.deepseek_provider', 'victor.providers.fireworks_provider', 'victor.providers.google_provider', 'victor.providers.groq_provider', 'victor.providers.huggingface_provider', 'victor.providers.inferflux_provider', 'victor.providers.llamacpp_provider', 'victor.providers.lmstudio_provider', 'victor.providers.mistral_provider', 'victor.providers.mlx_provider', 'victor.providers.moonshot_provider', 'victor.providers.ollama_provider', 'victor.providers.openai_provider', 'victor.providers.openrouter_provider', 'victor.providers.qwen_provider', 'victor.providers.replicate_provider', 'victor.providers.together_provider', 'victor.providers.vertex_provider', 'victor.providers.vllm_provider', 'victor.providers.xai_provider', 'victor.providers.zai_provider'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'PIL', 'cv2', 'numpy.distutils', 'pytest', 'sphinx', 'IPython', 'jupyter', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='victor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
