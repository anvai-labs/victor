#!/usr/bin/env python3
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

"""Example: LCEL-style tool composition in Victor.

This example demonstrates how to use the composition module to build
functional tool chains using LangChain Expression Language (LCEL) style
patterns with Victor's tools.

Features demonstrated:
1. Pipe chaining (`|` operator)
2. Parallel execution (RunnableParallel)
3. Conditional routing (RunnableBranch)
4. Result transformation (RunnableLambda)
5. Integration with Victor's @tool decorated functions
"""

import asyncio
from pathlib import Path
from typing import Any, Dict

# Import composition primitives
from victor.tools.composition import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    as_runnable,
    chain,
    extract_output,
    map_keys,
    parallel,
)

# Import actual Victor tools (decorated functions)
from victor.tools.filesystem import ls, read

# =============================================================================
# Example 1: Simple Pipe Chain
# =============================================================================


async def example_simple_chain():
    """Demonstrate basic pipe chaining."""
    print("\n=== Example 1: Simple Pipe Chain ===")

    # Create runnables from Victor tools
    ls_runnable = as_runnable(ls)
    read_runnable = as_runnable(read)

    # Define a transformer to extract file path from ls output
    def get_first_python_file(result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract first .py file from ls results."""
        if not result.get("success"):
            return {"path": None, "error": result.get("error")}

        output = result.get("output", {})
        items = output.get("items", [])

        for item in items:
            if item.get("type") == "file":
                name = item.get("name", item.get("path", ""))
                if name.endswith(".py"):
                    return {"path": item.get("full_path", name)}

        return {"path": None, "error": "No Python files found"}

    # Build chain: ls -> extract first .py file -> read
    simple_chain = ls_runnable | RunnableLambda(get_first_python_file) | read_runnable

    # Execute chain
    result = await simple_chain.invoke({"path": "victor/tools"})

    if result.get("success"):
        output = result.get("output", "")
        preview = output[:200] if isinstance(output, str) else str(output)[:200]
        print(f"Read file successfully. First 200 chars:\n{preview}...")
    else:
        print(f"Chain failed: {result.get('error')}")


# =============================================================================
# Example 2: Parallel Execution
# =============================================================================


async def example_parallel():
    """Demonstrate parallel tool execution."""
    print("\n=== Example 2: Parallel Execution ===")

    # Execute multiple directory listings in parallel
    parallel_ls = RunnableParallel(
        tools=as_runnable(ls).bind(path="victor/tools", depth=1),
        agent=as_runnable(ls).bind(path="victor/agent", depth=1),
        config=as_runnable(ls).bind(path="victor/config", depth=1),
    )

    result = await parallel_ls.invoke({})

    for name, data in result.items():
        if data.get("success"):
            output = data.get("output", {})
            item_count = output.get("count", 0)
            print(f"  {name}: {item_count} items")
        else:
            print(f"  {name}: Error - {data.get('error')}")


# =============================================================================
# Example 3: Conditional Routing
# =============================================================================


async def example_conditional():
    """Demonstrate conditional branching."""
    print("\n=== Example 3: Conditional Routing ===")

    # Define conditions based on file path
    def is_python_file(d: Dict) -> bool:
        path = d.get("path", "")
        return path.endswith(".py")

    def is_config_file(d: Dict) -> bool:
        path = d.get("path", "")
        return path.endswith((".yaml", ".yml", ".toml", ".json"))

    def is_markdown_file(d: Dict) -> bool:
        path = d.get("path", "")
        return path.endswith(".md")

    # Create branch with different processors
    python_processor = RunnableLambda(
        lambda d: {"file_type": "Python source", "path": d.get("path")}
    )
    config_processor = RunnableLambda(
        lambda d: {"file_type": "Configuration", "path": d.get("path")}
    )
    markdown_processor = RunnableLambda(
        lambda d: {"file_type": "Documentation", "path": d.get("path")}
    )
    default_processor = RunnableLambda(lambda d: {"file_type": "Unknown", "path": d.get("path")})

    branch = RunnableBranch(
        (is_python_file, python_processor),
        (is_config_file, config_processor),
        (is_markdown_file, markdown_processor),
        default=default_processor,
    )

    # Test with different file types
    test_files = [
        {"path": "main.py"},
        {"path": "config.yaml"},
        {"path": "README.md"},
        {"path": "data.bin"},
    ]

    for test_input in test_files:
        result = await branch.invoke(test_input)
        print(f"  {result['path']}: {result['file_type']}")


# =============================================================================
# Example 4: Complex Pipeline
# =============================================================================


async def example_complex_pipeline():
    """Demonstrate a complex multi-step pipeline."""
    print("\n=== Example 4: Complex Pipeline ===")

    # Step 1: List directory
    list_dir = as_runnable(ls)

    # Step 2: Filter to Python files
    def filter_python_files(result: Dict) -> Dict:
        if not result.get("success"):
            return {"files": [], "error": result.get("error")}

        output = result.get("output", {})
        items = output.get("items", [])

        py_files = [
            item.get("full_path", item.get("path", ""))
            for item in items
            if item.get("type") == "file"
            and (item.get("name", item.get("path", ""))).endswith(".py")
        ]
        return {"files": py_files[:3], "count": len(py_files)}  # Limit to 3

    # Step 3: Read each file in parallel
    def create_parallel_reads(result: Dict) -> Dict:
        """Prepare input for parallel file reads."""
        return {"files": result.get("files", []), "count": result.get("count", 0)}

    async def read_files_parallel(data: Dict) -> Dict:
        """Read multiple files in parallel."""
        files = data.get("files", [])
        if not files:
            return {"files_read": 0, "results": {}}

        read_runnable = as_runnable(read)
        results = {}

        for file_path in files:
            result = await read_runnable.invoke({"path": file_path})
            if result.get("success"):
                output = result.get("output", "")
                line_count = len(output.split("\n")) if isinstance(output, str) else 0
                results[file_path] = {"lines": line_count, "success": True}
            else:
                results[file_path] = {"error": result.get("error"), "success": False}

        return {"files_read": len(results), "results": results}

    # Build the pipeline
    pipeline = list_dir | RunnableLambda(filter_python_files) | RunnableLambda(read_files_parallel)

    # Execute
    result = await pipeline.invoke({"path": "victor/tools", "depth": 1})

    print(f"  Files read: {result.get('files_read', 0)}")
    for path, info in result.get("results", {}).items():
        if info.get("success"):
            print(f"    {Path(path).name}: {info['lines']} lines")
        else:
            print(f"    {Path(path).name}: Error - {info.get('error')}")


# =============================================================================
# Example 5: Passthrough for Preserving Data
# =============================================================================


async def example_passthrough():
    """Demonstrate using passthrough to preserve original data."""
    print("\n=== Example 5: Passthrough Pattern ===")

    # Create a parallel that preserves original input alongside transformed data
    enhanced = RunnableParallel(
        original=RunnablePassthrough(),
        enriched=RunnableLambda(
            lambda d: {
                **d,
                "processed": True,
                "timestamp": "2025-01-15T10:00:00Z",
            }
        ),
    )

    input_data = {"path": "/some/file.py", "content": "print('hello')"}
    result = await enhanced.invoke(input_data)

    print(f"  Original: {result['original']}")
    print(f"  Enriched: {result['enriched']}")


# =============================================================================
# Example 6: Using Helper Functions
# =============================================================================


async def example_helpers():
    """Demonstrate chain building helper functions."""
    print("\n=== Example 6: Helper Functions ===")

    # Using chain() helper
    c = chain(
        RunnableLambda(lambda x: x + 1),
        RunnableLambda(lambda x: x * 2),
        RunnableLambda(lambda x: {"result": x}),
    )
    result = await c.invoke(5)
    print(f"  chain(+1, *2): {result}")

    # Using parallel() helper
    p = parallel(
        doubled=RunnableLambda(lambda x: x * 2),
        squared=RunnableLambda(lambda x: x**2),
    )
    result = await p.invoke(5)
    print(f"  parallel(x2, x^2): {result}")

    # Using map_keys()
    mapper = map_keys({"old_name": "new_name"})
    result = mapper({"old_name": "value", "keep": "this"})
    print(f"  map_keys: {result}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Victor Tool Composition Examples (LCEL-style)")
    print("=" * 60)

    await example_simple_chain()
    await example_parallel()
    await example_conditional()
    await example_complex_pipeline()
    await example_passthrough()
    await example_helpers()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
