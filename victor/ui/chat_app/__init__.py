# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Victor web chat UI — a Chainlit surface bound to :class:`VictorClient`.

This is a pure-Python, agent-native chat surface (beyond the terminal). It binds to
``VictorClient.stream()`` in-process (no HTTP bridge) and renders Victor's event model
(CONTENT / THINKING / TOOL_CALL / TOOL_RESULT) as streamed tokens and collapsible steps.

Launch with ``victor ui`` (requires the optional ``chat-ui`` extra: ``pip install
'victor-ai[chat-ui]'``). The event→render mapping lives in :mod:`event_mapping` and is
Chainlit-free so it can be unit-tested without the UI dependency installed.
"""

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind, map_event

__all__ = ["RenderAction", "RenderKind", "map_event"]
