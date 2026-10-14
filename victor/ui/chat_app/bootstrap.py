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

"""Chainlit entrypoint: register Victor callbacks and packaged presentation assets."""

from chainlit.config import config
from chainlit.server import app

from victor.ui.chat_app import app as callbacks  # noqa: F401
from victor.ui.chat_app.presentation import configure_presentation

configure_presentation(app, config.ui, root_path=config.run.root_path)
