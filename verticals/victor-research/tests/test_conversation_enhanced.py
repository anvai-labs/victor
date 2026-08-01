# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the enhanced research conversation manager and context tracking."""

from victor_contracts import TurnType

from victor_research.conversation_enhanced import (
    EnhancedResearchConversationManager,
    ResearchContext,
)


class TestResearchContext:
    def test_add_research_question_deduplicates(self):
        ctx = ResearchContext()

        ctx.add_research_question("What drives LLM hallucination?")
        ctx.add_research_question("What drives LLM hallucination?")

        assert ctx.research_questions == ["What drives LLM hallucination?"]

    def test_add_hypothesis_with_status(self):
        ctx = ResearchContext()

        ctx.add_hypothesis("Retrieval reduces hallucination", status="testing")

        assert ctx.hypotheses == [
            {"statement": "Retrieval reduces hallucination", "status": "testing"}
        ]

    def test_add_hypothesis_defaults_untested(self):
        ctx = ResearchContext()

        ctx.add_hypothesis("Bigger models cite better")

        assert ctx.hypotheses[0]["status"] == "untested"

    def test_add_data_source(self):
        ctx = ResearchContext()

        ctx.add_data_source("arxiv:2401.00001", "paper")

        assert ctx.data_sources[0] == {"source": "arxiv:2401.00001", "type": "paper"}

    def test_add_experiment_and_finding(self):
        ctx = ResearchContext()

        ctx.add_experiment("ablation on retrieval depth", "depth 5 optimal")
        ctx.add_finding("citations improve with retrieval", category="methodological")

        assert ctx.experiments[0]["result"] == "depth 5 optimal"
        assert ctx.findings[0]["category"] == "methodological"

    def test_to_dict_round_trip(self):
        ctx = ResearchContext()
        ctx.add_research_question("Q1")

        data = ctx.to_dict()

        assert set(data) == {
            "research_questions",
            "hypotheses",
            "data_sources",
            "experiments",
            "findings",
            "references",
            "next_steps",
        }
        assert data["research_questions"] == ["Q1"]


class TestEnhancedConversationManager:
    def test_add_message_returns_turn_id(self):
        manager = EnhancedResearchConversationManager()

        turn_id = manager.add_message("user", "research AI safety trends", TurnType.USER)

        assert turn_id
        history = manager.get_history()
        assert any(m["content"] == "research AI safety trends" for m in history)

    def test_track_research_question_flows_into_context(self):
        manager = EnhancedResearchConversationManager()

        manager.track_research_question("How do agents plan?")

        ctx = manager.get_research_context()
        assert "How do agents plan?" in ctx.research_questions

    def test_track_data_source_and_finding(self):
        manager = EnhancedResearchConversationManager()
        manager.track_data_source("https://example.com/report", "web")
        manager.track_finding("agents plan hierarchically")

        ctx = manager.get_research_context()
        assert ctx.data_sources[0]["type"] == "web"
        assert ctx.findings[0]["finding"] == "agents plan hierarchically"

    def test_research_summary_reflects_tracked_work(self):
        manager = EnhancedResearchConversationManager()
        manager.track_research_question("What is RAG?")
        manager.track_hypothesis("RAG improves grounding", status="confirmed")

        summary = manager.get_research_summary()

        assert summary  # non-empty summary of research work

    def test_clear_history_resets_conversation(self):
        manager = EnhancedResearchConversationManager()
        manager.add_message("user", "hello", TurnType.USER)

        manager.clear_history()

        assert manager.get_history() == []

    def test_needs_summarization_threshold(self):
        manager = EnhancedResearchConversationManager(
            max_history_turns=10, summarization_threshold=2
        )

        assert manager.needs_summarization() is False
        manager.add_message("user", "one", TurnType.USER)
        manager.add_message("assistant", "two", TurnType.ASSISTANT)
        manager.add_message("user", "three", TurnType.USER)

        assert manager.needs_summarization() is True
