#!/bin/bash
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Prepare derived tool embeddings for the current installed tool registry.
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python3 <<'PYTHON'
import asyncio
import logging
import numpy as np
from victor.core.data_cache import load_validated_data
from victor.agent.shared_tool_registry import SharedToolRegistry
from victor.tools.registry import ToolRegistry
from victor.tools.semantic_selector import SemanticToolSelector

async def main():
    registry = ToolRegistry()
    for tool in SharedToolRegistry.get_instance().get_all_tools_for_registration():
        registry.register(tool)
    tools = registry.list_tools()
    if not tools:
        raise RuntimeError("No installed tools were discovered")
    selector = SemanticToolSelector(embedding_model="BAAI/bge-small-en-v1.5")
    try:
        await selector.initialize_tool_embeddings(registry)
        saved = load_validated_data(
            selector.cache_file, validators=[], logger=logging.getLogger(__name__),
            label="Prepared tool embeddings",
        )
        names = sorted(tool.name for tool in tools)
        if (
            saved is None
            or saved.get("tools_hash") != selector._calculate_tools_hash(registry)
            or saved.get("embedding_model") != selector.embedding_model
            or saved.get("cache_version") != selector.CACHE_VERSION
            or saved.get("tool_names") != names
            or saved.get("tool_count") != len(names)
            or set(saved.get("embeddings", {})) != set(names)
        ):
            raise RuntimeError("Current tool embeddings were not persisted")
        for vector in saved["embeddings"].values():
            if not isinstance(vector, np.ndarray) or vector.shape != (384,) or not np.isfinite(vector).all():
                raise RuntimeError("Persisted tool embedding is invalid")
        print(f"Prepared embeddings for {len(tools)} tools: {selector.cache_file}")
    finally:
        await selector.close()

asyncio.run(main())
PYTHON
