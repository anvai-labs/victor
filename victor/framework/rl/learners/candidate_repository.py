# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SQLite persistence for prompt candidates.

Extracted from ``prompt_optimizer.py`` (whose learner delegates to this) so the
``agent_prompt_candidate`` read/write/delete SQL lives in one place, apart from
the learner's in-memory index and evolution logic. The repository owns only a
database connection; the learner keeps the ``_candidates`` dict and all
lookup/selection behaviour.

Schema creation stays on the learner (``_ensure_tables``) because it runs as a
BaseLearner init hook, before this repository exists.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from victor.core.json_utils import json_dumps, json_loads
from victor.framework.rl.learners.candidate import PromptCandidate

logger = logging.getLogger(__name__)


def candidate_key(section_name: str, provider: str = "default") -> str:
    """Build the dict key for a (section, provider) pair."""
    return f"{section_name}::{provider}"


class CandidateRepository:
    """Reads, writes, and deletes prompt candidates in the project database.

    Candidates are keyed by ``(section_name, provider)`` for provider-aware
    prompt evolution; the returned dict uses ``"section_name::provider"`` keys
    (see :func:`candidate_key`).
    """

    def __init__(self, db: Any):
        self._db = db

    def load_all(self) -> Dict[str, List[PromptCandidate]]:
        """Load every persisted candidate, grouped by ``(section, provider)``."""
        from victor.core.schema import Tables

        candidates: Dict[str, List[PromptCandidate]] = {}
        try:
            cursor = self._db.execute(
                f"SELECT section_name, provider, text_hash, text, generation, parent_hash, "
                f"completion_score, token_efficiency, tool_effectiveness, "
                f"alpha, beta, sample_count, instance_scores, coverage_count, "
                f"is_on_frontier, char_length, benchmark_score, benchmark_runs, "
                f"benchmark_passed, is_active, strategy_name, strategy_chain, "
                f"requires_benchmark "
                f"FROM {Tables.AGENT_PROMPT_CANDIDATE}"
            )
            for row in cursor.fetchall():
                try:
                    instance_scores = json_loads(row[12] or "{}")
                except Exception:
                    instance_scores = {}
                candidate = PromptCandidate(
                    section_name=row[0],
                    provider=row[1] or "default",
                    text_hash=row[2],
                    text=row[3],
                    generation=row[4],
                    parent_hash=row[5] or "",
                    scores={
                        "completion_score": row[6],
                        "token_efficiency": row[7],
                        "tool_effectiveness": row[8],
                    },
                    alpha=row[9],
                    beta_val=row[10],
                    sample_count=row[11],
                    instance_scores=instance_scores,
                    coverage_count=row[13] or 0,
                    is_on_frontier=bool(row[14]),
                    char_length=row[15] or 0,
                    benchmark_score=row[16] or 0.0,
                    benchmark_runs=row[17] or 0,
                    benchmark_passed=bool(row[18]),
                    is_active=bool(row[19]),
                    strategy_name=row[20] or "gepa",
                    strategy_chain=row[21] or row[20] or "gepa",
                    requires_benchmark=bool(row[22]),
                )
                key = candidate_key(row[0], row[1] or "default")
                candidates.setdefault(key, []).append(candidate)
            total = sum(len(v) for v in candidates.values())
            if total:
                logger.info("Loaded %d prompt candidates from database", total)
        except Exception as e:
            logger.debug("Failed to load prompt candidates: %s", e)
        return candidates

    def save(self, candidate: PromptCandidate) -> None:
        """Persist a candidate to the database (insert or replace)."""
        from victor.core.schema import Tables

        try:
            self._db.execute(
                f"INSERT OR REPLACE INTO {Tables.AGENT_PROMPT_CANDIDATE} "
                f"(section_name, provider, text_hash, text, generation, parent_hash, "
                f"completion_score, token_efficiency, tool_effectiveness, "
                f"alpha, beta, sample_count, instance_scores, coverage_count, "
                f"is_on_frontier, char_length, benchmark_score, benchmark_runs, "
                f"benchmark_passed, is_active, strategy_name, strategy_chain, "
                f"requires_benchmark) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    candidate.section_name,
                    candidate.provider,
                    candidate.text_hash,
                    candidate.text,
                    candidate.generation,
                    candidate.parent_hash,
                    candidate.scores.get("completion_score", 0.0),
                    candidate.scores.get("token_efficiency", 0.0),
                    candidate.scores.get("tool_effectiveness", 0.0),
                    candidate.alpha,
                    candidate.beta_val,
                    candidate.sample_count,
                    json_dumps(candidate.instance_scores or {}),
                    candidate.coverage_count,
                    int(candidate.is_on_frontier),
                    candidate.char_length or len(candidate.text),
                    candidate.benchmark_score,
                    candidate.benchmark_runs,
                    int(candidate.benchmark_passed),
                    int(candidate.is_active),
                    candidate.strategy_name,
                    candidate.strategy_chain,
                    int(candidate.requires_benchmark),
                ),
            )
            self._db.commit()
        except Exception as e:
            logger.warning("Failed to save prompt candidate: %s", e)

    def delete_many(self, text_hashes: List[str]) -> None:
        """Delete candidates by text hash (used when pruning a section)."""
        from victor.core.schema import Tables

        if not text_hashes:
            return
        for text_hash in text_hashes:
            try:
                self._db.execute(
                    f"DELETE FROM {Tables.AGENT_PROMPT_CANDIDATE} WHERE text_hash = ?",
                    (text_hash,),
                )
            except Exception:
                pass
        self._db.commit()
