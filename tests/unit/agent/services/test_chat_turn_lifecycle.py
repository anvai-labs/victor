"""Turn-lifecycle capability contracts for ChatService runtime binding."""

from dataclasses import FrozenInstanceError

import pytest

from victor.agent.services.chat_turn_lifecycle import ChatTurnLifecycle


def test_lifecycle_keeps_setup_and_teardown_paired_and_immutable() -> None:
    def setup() -> None:
        return None

    def teardown() -> None:
        return None

    lifecycle = ChatTurnLifecycle(setup=setup, teardown=teardown)

    assert lifecycle.setup is setup
    assert lifecycle.teardown is teardown
    assert not hasattr(lifecycle, "__dict__")
    with pytest.raises(FrozenInstanceError):
        lifecycle.setup = None
