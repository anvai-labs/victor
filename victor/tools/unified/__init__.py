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

"""Bash-style unified command tools (`domain action args`).

This package hosts the command-shell tool dispatchers — ``shell``,
``git``, ``web``, ``code`` — each a ``@tool`` taking a single ``cmd: str``
parsed via :mod:`victor.tools.unified.parser`, plus their shared helpers
(``_search_helpers``, ``_vertical_resolver``).

The ``fs`` domain has been REMOVED. File operations are now first-class tools
with named parameters: ``read``, ``edit``, ``write`` (from
``victor.tools.filesystem`` and ``victor.tools.file_editor_tool``).

Tool discovery and execution do **not** live here. Discovery is owned by
:class:`victor.agent.shared_tool_registry.SharedToolRegistry` (process-wide
cache) and per-session execution by :class:`victor.tools.registry.ToolRegistry`;
selection by :mod:`victor.tools.unified`-independent
:mod:`victor.agent.tool_selection`. The former ``UnifiedToolRegistry`` /
adapters that lived here were dead code and have been removed.
"""

# NOTE: The bash-style command tools (code_tool, git_tool, search_tool,
# shell_tool, web_tool) are intentionally NOT re-exported here. Importing them as
# ``from victor.tools.unified.<name>_tool import <name>_tool`` rebinds the
# submodule name to the function and shadows the module object. That breaks
# ``mock.patch("victor.tools.unified.<name>_tool.<func>")`` on Python 3.10,
# whose ``mock`` dotted resolver does not fall back to ``sys.modules``. The
# canonical import path
# ``from victor.tools.unified.<name>_tool import <name>_tool`` remains the
# supported way to access these tools.
