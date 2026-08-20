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

"""Demo of Git tool with unified operations.

This demonstrates:
1. Git status and diff operations
2. File staging
3. Commit creation
4. Branch management
5. Commit history (log)
6. Conflict analysis

Usage:
    python examples/git_tool_demo.py
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from victor.tools.git_tool import git, conflicts


def run_command(cmd: str, cwd: Path = None) -> str:
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout


async def main():
    """Demo git tool operations."""
    print("🎯 Git Tool Demo")
    print("=" * 70)
    print("\nDemonstrating unified git operations\n")

    # Create temporary git repo for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Initialize git repo
        print("1️⃣ Setting up demo repository...")
        print("-" * 70)
        run_command("git init", tmpdir)
        run_command("git config user.email 'demo@example.com'", tmpdir)
        run_command("git config user.name 'Demo User'", tmpdir)

        # Change to temp directory for git operations
        original_dir = os.getcwd()
        os.chdir(tmpdir)

        try:
            # Create some files
            (tmpdir / "README.md").write_text("""# Demo Project

This is a demo project for testing Victor's git tool.

## Features
- Git integration
- AI-powered commit messages
- Smart operations
""")

            (tmpdir / "main.py").write_text("""def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")

            # Test 1: Git status
            print("\n2️⃣ Git Status")
            print("-" * 70)
            result = await git(operation="status")
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 2: Stage files
            print("\n3️⃣ Staging Files")
            print("-" * 70)
            result = await git(operation="stage", files=["README.md", "main.py"])
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 3: Show diff
            print("\n4️⃣ Staged Diff")
            print("-" * 70)
            result = await git(operation="diff", staged=True)
            if result["success"]:
                output = result["output"]
                print(output[:500] + "..." if len(output) > 500 else output)
            else:
                print(f"Error: {result['error']}")

            # Test 4: Commit
            print("\n5️⃣ Commit Changes")
            print("-" * 70)
            result = await git(
                operation="commit",
                message="Initial commit: Add README and main.py\n\nThis is a demo commit showing Victor's git tool capabilities.",
            )
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 5: Git log
            print("\n6️⃣ Commit History")
            print("-" * 70)
            result = await git(operation="log", limit=5)
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 6: Create and switch branch
            print("\n7️⃣ Branch Operations")
            print("-" * 70)
            result = await git(operation="branch", branch="feature/add-tests")
            if result["success"]:
                print("Created/switched to branch: feature/add-tests")
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 7: Make more changes
            print("\n8️⃣ Making More Changes")
            print("-" * 70)
            (tmpdir / "tests.py").write_text("""import pytest

def test_main():
    assert True
""")

            result = await git(operation="stage", files=["tests.py"])
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 8: Commit with manual message
            result = await git(operation="commit", message="test: Add basic test file")
            print(f"\nCommit result: {result['output'] if result['success'] else result['error']}")

            # Test 9: Switch back to main
            print("\n9️⃣ Switching Branches")
            print("-" * 70)
            result = await git(operation="branch", branch="master")
            if result["success"]:
                print(result["output"])
            else:
                # Try 'main' if 'master' doesn't exist
                result = await git(operation="branch", branch="main")
                if result["success"]:
                    print(result["output"])
                else:
                    print(f"Note: {result['error']}")

            # Test 10: List all branches
            print("\n🔟 List All Branches")
            print("-" * 70)
            result = await git(operation="branch")
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 11: Make changes on main
            print("\n1️⃣1️⃣ Making Changes on Main Branch")
            print("-" * 70)
            (tmpdir / "utils.py").write_text("""def helper():
    return "Helper function"
""")

            result = await git(operation="stage")
            result = await git(operation="commit", message="feat: Add utility functions")
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 12: Final log showing multiple commits
            print("\n1️⃣2️⃣ Final Commit History")
            print("-" * 70)
            result = await git(operation="log", limit=10)
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            # Test 13: Conflict analysis (no conflicts expected)
            print("\n1️⃣3️⃣ Conflict Analysis")
            print("-" * 70)
            result = await conflicts()
            if result["success"]:
                print(result["output"])
            else:
                print(f"Error: {result['error']}")

            print("\n\n✨ Demo Complete!")
            print("\nGit Tool Features Demonstrated:")
            print("  ✓ Repository status checking")
            print("  ✓ File staging (individual and bulk)")
            print("  ✓ Diff viewing (staged and unstaged)")
            print("  ✓ Committing changes")
            print("  ✓ Viewing commit history")
            print("  ✓ Branch creation and switching")
            print("  ✓ Multiple commits and branch management")
            print("  ✓ Conflict analysis")

            print("\n\n🤖 With AI Provider Available:")
            print("  • AI-generated commit messages from diff analysis (git_suggest_commit)")
            print("  • Auto-generated PR titles and descriptions (git_create_pr)")
            print("  • Conflict resolution suggestions (conflicts)")

            print("\n\n📚 API Examples:")
            print("""
# Status
result = await git(operation="status")

# Stage all changes
result = await git(operation="stage")

# Stage specific files
result = await git(operation="stage", files=["file1.py", "file2.py"])

# Show diff
result = await git(operation="diff")

# Show staged diff
result = await git(operation="diff", staged=True)

# Commit
result = await git(operation="commit", message="feat: Add feature")

# View log
result = await git(operation="log", limit=10)

# List branches
result = await git(operation="branch")

# Create/switch branch
result = await git(operation="branch", branch="feature/new-feature")
""")

        finally:
            # Restore original directory
            os.chdir(original_dir)


if __name__ == "__main__":
    asyncio.run(main())
