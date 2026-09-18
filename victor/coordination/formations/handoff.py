"""HandoffFormation: bounded transcript execution; durable partial resume unsupported."""

from victor.coordination.formations.conversation import ConversationFormation


class HandoffFormation(ConversationFormation):
    mode = "handoff"

    def supports_durable_pause(self) -> bool:
        """Approvals remain inline; transcript partial resume is unsupported."""
        return False
