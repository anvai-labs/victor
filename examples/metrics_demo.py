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

"""Demo of Victor's Code Metrics Tool.

Usage:
    python examples/metrics_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from victor.tools.metrics_tool import metrics


async def main():
    """Run metrics demos."""
    print("🎯 Victor Code Metrics Tool Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create demo file with varying complexity
        demo_code = '''
def simple_function(x):
    return x * 2

def complex_function(data, threshold):
    # TODO: Optimize this function
    result = []
    for item in data:
        if item > threshold:
            if item % 2 == 0:
                for i in range(item):
                    if i % 3 == 0:
                        result.append(i)
                    elif i % 5 == 0:
                        result.append(i * 2)
                    else:
                        result.append(i + 1)
    # FIXME: Handle edge cases
    return result

def moderate_function(values):
    """Process values."""
    if not values:
        return []
    return [v for v in values if v > 0]
'''

        file_path = Path(temp_dir) / "code.py"
        file_path.write_text(demo_code)

        print("\n📊 Complexity Analysis")
        print("=" * 70)
        result = await metrics(
            path=str(file_path),
            metrics_list=["complexity"],
        )
        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        print("\n\n📈 Maintainability Index")
        print("=" * 70)
        result = await metrics(
            path=str(file_path),
            metrics_list=["maintainability"],
        )
        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        print("\n\n💰 Technical Debt Analysis")
        print("=" * 70)
        result = await metrics(
            path=str(file_path),
            metrics_list=["debt"],
        )
        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        print("\n\n📋 Comprehensive Analysis")
        print("=" * 70)
        result = await metrics(
            path=str(file_path),
            metrics_list=["all"],
        )
        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        print("\n\n✨ Demo Complete!")
        print("\nVictor's Metrics Tool provides:")
        print("  • Cyclomatic complexity analysis")
        print("  • Maintainability index calculation")
        print("  • Lines of code metrics")
        print("  • Code quality insights")


if __name__ == "__main__":
    asyncio.run(main())
