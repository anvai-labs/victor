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

"""Switch command — deprecated, now handled by unified /model command.

The /switch command is absorbed into /model via the "switch" alias.
This file is kept for backward compatibility but the SwitchCommand class
is no longer registered. The ModelCommand in model.py handles all
switching functionality including --resume support.
"""

# NOTE: SwitchCommand is no longer registered.
# /switch is handled by ModelCommand via aliases=["models", "switch"]
# in victor/ui/slash/commands/model.py

__all__: list[str] = []
