from __future__ import annotations

import pytest

from victor.agent.conversation.store import ConversationStore
from victor.providers.base import BaseProvider, CompletionResponse, Message, StreamChunk


class DummyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "dummy"

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ) -> CompletionResponse:
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ):
        if False:
            yield StreamChunk()
        raise NotImplementedError

    async def close(self) -> None:
        return None


def test_conversation_store_uses_fast_native_tokenizer(monkeypatch, tmp_path):
    """Test ConversationStore uses fast native tokenizer.

    Note: Tests use temporary project.db for isolation.
    In production, ConversationStore uses project.db (consolidated database).
    """
    calls: list[str] = []

    def fake_count_tokens_fast(text: str) -> int:
        calls.append(text)
        return 9

    monkeypatch.setattr(
        "victor.processing.native.tokenizer.count_tokens_fast",
        fake_count_tokens_fast,
    )

    store = ConversationStore(db_path=tmp_path / "project.db")

    assert store._estimate_tokens("summarize this diff") == 9
    assert calls == ["summarize this diff"]


@pytest.mark.asyncio
async def test_base_provider_uses_fast_native_tokenizer(monkeypatch):
    calls: list[str] = []

    def fake_count_tokens_fast(text: str) -> int:
        calls.append(text)
        return 23

    monkeypatch.setattr(
        "victor.processing.native.tokenizer.count_tokens_fast",
        fake_count_tokens_fast,
    )

    provider = DummyProvider()

    assert await provider.count_tokens("provider text") == 23
    assert calls == ["provider text"]
