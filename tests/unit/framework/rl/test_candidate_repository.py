# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Characterization tests for the extracted CandidateRepository.

These pin the save -> load round-trip (every persisted field survives) and the
delete path that moved out of PromptOptimizerLearner.
"""

import sqlite3

import pytest

from victor.framework.rl.learners.candidate import PromptCandidate
from victor.framework.rl.learners.candidate_repository import CandidateRepository, candidate_key
from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner


@pytest.fixture
def repo():
    """A repository backed by a learner-initialized DB (schema created via the
    BaseLearner _ensure_tables hook)."""
    db = sqlite3.connect(":memory:")
    learner = PromptOptimizerLearner(name="fixture", db_connection=db)
    return learner._candidate_repo


def _candidate(**kw):
    base = {
        "section_name": "GROUNDING_RULES",
        "provider": "ollama",
        "text": "guidance text",
        "text_hash": "h1",
        "generation": 3,
        "parent_hash": "p0",
    }
    base.update(kw)
    return PromptCandidate(**base)


def test_candidate_key_format():
    assert candidate_key("SECT", "ollama") == "SECT::ollama"
    assert candidate_key("SECT") == "SECT::default"


class TestRoundTrip:
    def test_all_fields_survive_save_load(self, repo):
        c = _candidate(
            alpha=5.0,
            beta_val=2.0,
            sample_count=7,
            scores={"completion_score": 0.8, "token_efficiency": 0.6, "tool_effectiveness": 0.9},
            benchmark_score=0.75,
            benchmark_runs=4,
            benchmark_passed=True,
            is_active=True,
            strategy_name="prefpo",
            strategy_chain="gepa,prefpo",
            requires_benchmark=True,
            instance_scores={"i1": 0.5},
            coverage_count=2,
        )
        repo.save(c)

        loaded = repo.load_all()[candidate_key("GROUNDING_RULES", "ollama")][0]
        assert loaded.text == "guidance text"
        assert loaded.generation == 3
        assert loaded.alpha == 5.0 and loaded.beta_val == 2.0 and loaded.sample_count == 7
        assert loaded.scores["completion_score"] == 0.8
        assert loaded.benchmark_score == 0.75 and loaded.benchmark_runs == 4
        assert loaded.benchmark_passed is True and loaded.is_active is True
        assert loaded.strategy_name == "prefpo" and loaded.strategy_chain == "gepa,prefpo"
        assert loaded.requires_benchmark is True
        assert loaded.instance_scores == {"i1": 0.5}
        assert loaded.coverage_count == 2

    def test_insert_or_replace_updates_in_place(self, repo):
        repo.save(_candidate(text="v1", sample_count=1))
        repo.save(_candidate(text="v2", sample_count=9))  # same text_hash -> replace
        pool = repo.load_all()[candidate_key("GROUNDING_RULES", "ollama")]
        assert len(pool) == 1
        assert pool[0].text == "v2" and pool[0].sample_count == 9

    def test_default_provider_key(self, repo):
        repo.save(_candidate(provider="default", text_hash="hd"))
        assert candidate_key("GROUNDING_RULES", "default") in repo.load_all()


class TestDelete:
    def test_delete_many_removes_rows(self, repo):
        repo.save(_candidate(text_hash="keep"))
        repo.save(_candidate(text_hash="drop", provider="xai"))
        repo.delete_many(["drop"])
        loaded = repo.load_all()
        assert candidate_key("GROUNDING_RULES", "ollama") in loaded
        assert candidate_key("GROUNDING_RULES", "xai") not in loaded

    def test_delete_many_empty_is_noop(self, repo):
        repo.save(_candidate(text_hash="keep"))
        repo.delete_many([])
        assert repo.load_all()[candidate_key("GROUNDING_RULES", "ollama")]
