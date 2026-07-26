# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Cross-visit resume: ChatService.resume_session hydrates BOTH message stores.

The framework keeps two message stores that reset_conversation clears
separately — the ContextService buffer and the ConversationController history
(the streaming turn loop reads the latter). A correct resume must repopulate
both, or the agent 'looks resumed' but does not recall the prior turns through
the executor. These tests pin that with real Context/Controller objects.
"""

from __future__ import annotations

from victor.agent.conversation.controller import ConversationController
from victor.agent.message_history import MessageHistory
from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.agent.services.context_service import ContextService, ContextServiceConfig
from victor.providers.base import Message


def _stored_session():
    """A stored-session dict as load_session returns it: 2 prior turns."""
    history = MessageHistory(system_prompt="You are Victor.")
    history.add_user_message("what is 2+2?")
    history.add_assistant_message("4")
    return {
        "metadata": {"title": "arithmetic", "message_count": 2},
        "conversation": history.to_dict(),
    }


class _Noop:
    """Minimal stand-in for the services ChatService needs but resume ignores."""

    def is_healthy(self):
        return True


class TestControllerLoadHistory:
    def test_load_history_repoints_and_ensures_system(self):
        controller = ConversationController()
        controller.set_system_prompt("You are Victor.")

        history = MessageHistory(system_prompt="You are Victor.")
        history.add_user_message("hello")
        history.add_assistant_message("hi there")

        controller.load_history(history)

        # The controller now operates on the loaded history...
        assert controller.messages[-1].content == "hi there"
        # ...with a well-formed system message at the head.
        assert controller.messages[0].role == "system"

    def test_load_history_preserves_existing_system_head(self):
        controller = ConversationController()
        controller.set_system_prompt("New prompt.")
        history = MessageHistory()
        history._messages.append(Message(role="system", content="Stored prompt."))
        history._messages.append(Message(role="user", content="q"))

        controller.load_history(history)

        # A stored system head is not duplicated.
        system_count = sum(1 for m in controller.messages if m.role == "system")
        assert system_count == 1
        assert controller.messages[0].content == "Stored prompt."


class TestChatServiceResumeSession:
    def _service(self):
        context = ContextService(ContextServiceConfig())
        controller = ConversationController()
        controller.set_system_prompt("You are Victor.")
        service = ChatService(
            config=ChatServiceConfig(),
            provider_service=_Noop(),
            tool_service=_Noop(),
            context_service=context,
            recovery_service=_Noop(),
            conversation_controller=controller,
            streaming_coordinator=_Noop(),
        )
        return service, context, controller

    def test_resume_hydrates_both_stores(self):
        service, context, controller = self._service()

        metadata = service.resume_session(_stored_session())

        # Metadata is returned for the UI.
        assert metadata["title"] == "arithmetic"

        # BOTH stores carry the prior turns — the whole point.
        ctx_contents = [context._message_value(m, "content", "") for m in context.get_messages()]
        assert "what is 2+2?" in ctx_contents
        assert "4" in ctx_contents

        ctrl_contents = [m.content for m in controller.messages]
        assert "what is 2+2?" in ctrl_contents
        assert "4" in ctrl_contents

    def test_resume_replaces_prior_context(self):
        service, context, controller = self._service()
        # Seed some stale live context first.
        context.add_message(Message(role="user", content="stale"))
        controller.add_user_message("stale")

        service.resume_session(_stored_session())

        ctx_contents = [context._message_value(m, "content", "") for m in context.get_messages()]
        ctrl_contents = [m.content for m in controller.messages]
        assert "stale" not in ctx_contents
        assert "stale" not in ctrl_contents
        assert "4" in ctx_contents and "4" in ctrl_contents

    def test_resume_empty_conversation_is_safe(self):
        service, context, controller = self._service()
        metadata = service.resume_session({"metadata": {}, "conversation": {}})
        assert metadata == {}
