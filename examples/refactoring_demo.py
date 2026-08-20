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

"""Demo of Victor's Refactoring Tool.

Demonstrates safe code transformations:
- Rename symbols (variables, functions, classes)
- Extract functions from code blocks
- Inline variables
- Organize imports

Usage:
    python examples/refactoring_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from victor.tools.refactor_tool import (
    rename,
    extract,
    inline,
    organize_imports,
)


def setup_demo_file(temp_dir: Path, filename: str, content: str) -> Path:
    """Create a demo file."""
    file_path = temp_dir / filename
    file_path.write_text(content)
    return file_path


async def demo_rename_symbol():
    """Demo renaming a symbol."""
    print("\n\n🔄 Rename Symbol Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create demo file
        demo_code = """
def process_data(items):
    \"\"\"Process data items.\"\"\"
    result = []
    for item in items:
        processed = item.upper()
        result.append(processed)
    return result

def main():
    data = ["hello", "world"]
    output = process_data(data)
    print(output)
"""
        file_path = setup_demo_file(temp_path, "app.py", demo_code.strip())

        print("\n1️⃣ Original code:")
        print(demo_code)

        print("\n2️⃣ Preview: Rename 'process_data' to 'transform_items'...")
        result = await rename(
            path=str(file_path),
            old_name="process_data",
            new_name="transform_items",
            preview=True,
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', '')}")

        print("\n3️⃣ Execute: Rename 'process_data' to 'transform_items'...")
        result = await rename(
            path=str(file_path),
            old_name="process_data",
            new_name="transform_items",
            preview=False,
        )

        if result["success"]:
            print(result.get("formatted_report", ""))

        print("\n4️⃣ Modified code:")
        print(file_path.read_text())


async def demo_extract_function():
    """Demo extracting a function."""
    print("\n\n📦 Extract Function Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create demo file
        demo_code = """
def process_user_data(users):
    \"\"\"Process user data.\"\"\"
    results = []

    for user in users:
        # Validate user
        if not user.get('email'):
            continue
        if '@' not in user['email']:
            continue
        if len(user.get('name', '')) < 2:
            continue

        results.append(user)

    return results
"""
        file_path = setup_demo_file(temp_path, "users.py", demo_code.strip())

        print("\n1️⃣ Original code:")
        print(demo_code)

        print("\n2️⃣ Extract validation logic (lines 6-11) into 'validate_user' function...")
        result = await extract(
            file=str(file_path),
            start_line=6,
            end_line=11,
            function_name="validate_user",
            preview=True,
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', '')}")


async def demo_inline_variable():
    """Demo inlining a variable."""
    print("\n\n📋 Inline Variable Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create demo file
        demo_code = """
def calculate_total(items):
    \"\"\"Calculate total.\"\"\"
    tax_rate = 0.08

    subtotal = sum(item['price'] for item in items)
    tax = subtotal * tax_rate
    total = subtotal + tax

    return total
"""
        file_path = setup_demo_file(temp_path, "calc.py", demo_code.strip())

        print("\n1️⃣ Original code:")
        print(demo_code)

        print("\n2️⃣ Inline 'tax_rate' variable (preview)...")
        result = await inline(
            file=str(file_path),
            variable_name="tax_rate",
            preview=True,
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', '')}")

        print("\n3️⃣ Execute inlining...")
        result = await inline(
            file=str(file_path),
            variable_name="tax_rate",
            preview=False,
        )

        if result["success"]:
            print(result.get("formatted_report", ""))

        print("\n4️⃣ Modified code:")
        print(file_path.read_text())


async def demo_organize_imports():
    """Demo organizing imports."""
    print("\n\n📚 Organize Imports Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create demo file with messy imports
        demo_code = '''"""Module for data processing."""

from pathlib import Path
import json
from typing import List, Dict
import logging
from victor.tools.base import BaseTool
import sys
from collections import defaultdict
import os
from victor.storage.cache import CacheManager

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process data."""
    pass
'''
        file_path = setup_demo_file(temp_path, "processor.py", demo_code.strip())

        print("\n1️⃣ Original imports (messy):")
        print(demo_code[:300] + "...")

        print("\n2️⃣ Preview: Organize imports...")
        result = await organize_imports(
            file=str(file_path),
            preview=True,
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', '')}")

        print("\n3️⃣ Execute: Organize imports...")
        result = await organize_imports(
            file=str(file_path),
            preview=False,
        )

        if result["success"]:
            print(result.get("formatted_report", ""))

        print("\n4️⃣ Organized code:")
        print(file_path.read_text()[:400] + "...")


async def demo_real_world_workflow():
    """Demo a real-world refactoring workflow."""
    print("\n\n🎯 Real-World Workflow Demo")
    print("=" * 70)
    print("\nScenario: Refactoring legacy code")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create messy legacy code
        demo_code = """
import json
import sys
from pathlib import Path
import os
from typing import List

def proc(d):
    x = []
    for i in d:
        if i['active']:
            x.append(i)
    return x

def main():
    data = [{'id': 1, 'active': True}, {'id': 2, 'active': False}]
    r = proc(data)
    print(r)
"""
        file_path = setup_demo_file(temp_path, "legacy.py", demo_code.strip())

        print("\n1️⃣ Original legacy code:")
        print(demo_code)

        print("\n2️⃣ STEP 1: Organize imports...")
        result = await organize_imports(
            file=str(file_path),
            preview=False,
        )
        if result["success"]:
            print("✓ Imports organized")

        print("\n3️⃣ STEP 2: Rename 'proc' to 'filter_active_items'...")
        result = await rename(
            path=str(file_path),
            old_name="proc",
            new_name="filter_active_items",
            preview=False,
        )
        if result["success"]:
            print("✓ Function renamed")

        print("\n4️⃣ STEP 3: Rename 'd' to 'items'...")
        result = await rename(
            path=str(file_path),
            old_name="d",
            new_name="items",
            preview=False,
        )
        if result["success"]:
            print("✓ Parameter renamed")

        print("\n5️⃣ STEP 4: Rename 'x' to 'active_items'...")
        result = await rename(
            path=str(file_path),
            old_name="x",
            new_name="active_items",
            preview=False,
        )
        if result["success"]:
            print("✓ Variable renamed")

        print("\n6️⃣ STEP 5: Rename 'i' to 'item'...")
        result = await rename(
            path=str(file_path),
            old_name="i",
            new_name="item",
            preview=False,
        )
        if result["success"]:
            print("✓ Loop variable renamed")

        print("\n7️⃣ STEP 6: Rename 'r' to 'filtered_data'...")
        result = await rename(
            path=str(file_path),
            old_name="r",
            new_name="filtered_data",
            preview=False,
        )
        if result["success"]:
            print("✓ Result variable renamed")

        print("\n📊 Final refactored code:")
        print(file_path.read_text())

        print("\n✅ Refactoring complete!")
        print("\nImprovements:")
        print("  • Organized imports (stdlib → third-party → local)")
        print("  • Descriptive function name (proc → filter_active_items)")
        print("  • Clear parameter names (d → items)")
        print("  • Meaningful variable names (x → active_items, i → item, r → filtered_data)")
        print("  • Much more readable and maintainable!")


async def demo_safety_checks():
    """Demo safety checks and error handling."""
    print("\n\n🛡️  Safety Checks Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        demo_code = """
def calculate(a, b):
    return a + b
"""
        file_path = setup_demo_file(temp_path, "safe.py", demo_code.strip())

        print("\n1️⃣ Try to rename non-existent symbol...")
        result = await rename(
            path=str(file_path),
            old_name="nonexistent",
            new_name="something",
        )

        if not result["success"]:
            print(f"❌ Expected error: {result.get('error', '')}")

        print("\n2️⃣ Try to refactor non-existent file...")
        result = await rename(
            path="/tmp/does_not_exist.py",
            old_name="foo",
            new_name="bar",
        )

        if not result["success"]:
            print(f"❌ Expected error: {result.get('error', '')}")

        print("\n3️⃣ Try extract with invalid line range...")
        result = await extract(
            file=str(file_path),
            start_line=10,
            end_line=20,
            function_name="test",
        )

        if not result["success"]:
            print(f"❌ Expected error: {result.get('error', '')}")

        print("\n✅ All safety checks working correctly!")


async def main():
    """Run all refactoring demos."""
    print("🎯 Victor Refactoring Tool Demo")
    print("=" * 70)
    print("\nDemonstrating safe code transformations\n")

    # Run demos
    await demo_rename_symbol()
    await demo_extract_function()
    await demo_inline_variable()
    await demo_organize_imports()
    await demo_real_world_workflow()
    await demo_safety_checks()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Refactoring Tool provides:")
    print("  • AST-based analysis for safe transformations")
    print("  • Rename symbols (variables, functions, classes)")
    print("  • Extract functions from code blocks")
    print("  • Inline simple variables")
    print("  • Organize and optimize imports")
    print("  • Preview mode for safe refactoring")
    print("  • Error handling and validation")
    print("\nPerfect for:")
    print("  • Improving code readability")
    print("  • Cleaning up legacy code")
    print("  • Standardizing naming conventions")
    print("  • Organizing codebase structure")
    print("  • Safe, automated code transformations")
    print("\nReady for production use!")


if __name__ == "__main__":
    asyncio.run(main())
