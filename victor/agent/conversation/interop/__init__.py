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

"""Session transcript interop: claude ⇄ codex ⇄ victor, any pair.

Canonical-IR design: every format has a reader (format → ConversationSession)
and/or a writer (ConversationSession → format); conversions between any two
external tools route through the victor IR instead of N×N converters.

Public API:
    import_transcript(path, fmt=None) -> ImportedSession
    export_session(session, fmt, out, ...) -> Path
    install_session(session, target, ...) -> Path
    detect_format(path) -> DetectionResult
"""

from victor.agent.conversation.interop.formats import (
    DetectionResult,
    TranscriptFormat,
    UnsupportedFormatError,
    detect_format,
)
from victor.agent.conversation.interop.readers import ImportedSession, import_transcript
from victor.agent.conversation.interop.report import ConversionReport
from victor.agent.conversation.interop.writers import (
    export_session,
    install_session,
    to_persist_messages,
)
from victor.agent.conversation.session_model import ConversationSession

__all__ = [
    "ConversationSession",
    "ConversionReport",
    "DetectionResult",
    "ImportedSession",
    "TranscriptFormat",
    "UnsupportedFormatError",
    "detect_format",
    "export_session",
    "import_transcript",
    "install_session",
    "to_persist_messages",
]
