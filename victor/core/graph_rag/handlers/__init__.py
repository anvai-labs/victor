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

"""Language-specific edge detection handlers.

MIGRATION COMPLETE:
All edge detection has been migrated to victor_coding language plugins.
This package is now empty - use victor_coding.plugins for language-specific
edge detection implementations.

The victor-ai core now discovers and uses victor_coding plugins via:
- victor.core.graph_rag.language_handlers._get_victor_coding_handler()
- victor_coding.languages.registry.get_plugin_by_language()
"""

__all__ = []
