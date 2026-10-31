"""Default pins for conversational workflow nodes.

Added with the 5x'd stamina budgets (2026-10-31): the conversation turn cap
default lives on ``ConversationalNodeConfig`` and is pinned here.
"""

from victor.framework.conversations.node import ConversationalNodeConfig


def test_conversational_node_config_default_max_turns():
    """Conversational nodes default to the 5x'd turn cap (was 10)."""
    config = ConversationalNodeConfig()
    # 5x'd 2026-10-31: agents run longer autonomous sessions
    assert config.max_turns == 50
