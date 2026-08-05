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

"""Source code skeleton extraction.

Returns function/class signatures + docstrings without implementation details.
Useful for generating compact code overviews for LLM context windows.

Inspired by arXiv:2604.07502 (SE Conventions for Agentic Dev).
"""

from typing import List, Optional


def extract_skeleton(source: str, language: str = "python") -> str:
    """Extract a program skeleton from source code.

    Returns function/class signatures + docstrings without implementation
    details. Inspired by arXiv:2604.07502 (SE Conventions for Agentic Dev).

    Args:
        source: Source code string
        language: Programming language (currently 'python' supported)

    Returns:
        Skeleton string with signatures and docstrings
    """
    if not source.strip():
        return ""

    if language != "python":
        # Fallback: first 50 lines for unknown languages
        lines = source.split("\n")[:50]
        return "\n".join(lines)

    lines = source.split("\n")
    skeleton_lines: List[str] = []
    in_body = False
    in_docstring = False
    docstring_quote: Optional[str] = None

    for line in lines:
        stripped = line.strip()

        # Import statements — always include
        if stripped.startswith(("import ", "from ")):
            skeleton_lines.append(line)
            in_body = False
            continue

        # Decorators — include
        if stripped.startswith("@"):
            skeleton_lines.append(line)
            in_body = False
            continue

        # Class/function definitions — always include
        if stripped.startswith(("def ", "class ", "async def ")):
            skeleton_lines.append(line)
            in_body = True
            in_docstring = False
            continue

        # Docstrings right after def/class — include
        if in_body and not in_docstring:
            if '"""' in stripped or "'''" in stripped:
                skeleton_lines.append(line)
                quote = '"""' if '"""' in stripped else "'''"
                # Check if single-line docstring
                if stripped.count(quote) >= 2:
                    in_body = True  # Continue looking for nested defs
                    continue
                in_docstring = True
                docstring_quote = quote
                continue
            elif stripped.startswith("#"):
                # Comment right after def — include as pseudo-docstring
                skeleton_lines.append(line)
                continue
            else:
                # First non-docstring line in body — skip body
                in_body = True
                continue

        # Inside docstring — include until closing
        if in_docstring:
            skeleton_lines.append(line)
            if docstring_quote and docstring_quote in stripped:
                in_docstring = False
            continue

        # Module-level assignments/constants (not indented) — include
        if not stripped.startswith(" ") and not stripped.startswith("\t"):
            if "=" in stripped and not stripped.startswith("#"):
                # Module-level constant
                skeleton_lines.append(line)
                in_body = False
                continue

        # Blank lines between definitions — preserve structure
        if not stripped and not in_body:
            skeleton_lines.append(line)

    return "\n".join(skeleton_lines).rstrip()


__all__ = ["extract_skeleton"]
