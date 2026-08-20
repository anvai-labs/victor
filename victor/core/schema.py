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

"""Database schema constants and definitions for Victor.

This module provides centralized table name constants and schema definitions
to ensure consistency across the codebase. All database operations should
reference these constants instead of hardcoding table names.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    TABLE NAMING CONVENTION                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Domain Prefix:                                                      │
    │  ├── rl_     → Reinforcement Learning tables                        │
    │  ├── agent_  → Agent execution & teams                              │
    │  ├── ui_     → User interface (sessions, preferences)               │
    │  └── sys_    → System metadata                                      │
    │                                                                      │
    │  Naming Pattern: {domain}_{entity}                                   │
    │  Examples: rl_outcome, agent_team_run, ui_session                   │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from victor.core.schema import Tables, Schema

    # Reference table names
    cursor.execute(f"SELECT * FROM {Tables.RL_OUTCOME}")

    # Get full schema SQL
    schema_sql = Schema.RL_OUTCOME
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Tables:
    """Centralized table name constants.

    All table names are defined here to ensure consistency and enable
    easy refactoring. Use these constants instead of hardcoding table names.

    Naming Convention:
    - Domain prefix: rl_, agent_, ui_, sys_
    - Singular nouns: rl_outcome not rl_outcomes
    - Max 3 words: concise and descriptive
    """

    # ===========================================
    # SYSTEM DOMAIN (sys_)
    # ===========================================
    SYS_METADATA = "sys_metadata"
    SYS_SCHEMA_VERSION = "sys_schema_version"

    # ===========================================
    # REINFORCEMENT LEARNING DOMAIN (rl_)
    # ===========================================

    # Core RL tables (consolidated)
    RL_LEARNER = "rl_learner"  # Learner registry/configuration
    RL_OUTCOME = "rl_outcome"  # All learner outcomes (central fact table)
    RL_METRIC = "rl_metric"  # Telemetry and monitoring
    RL_Q_VALUE = "rl_q_value"  # Unified Q-values (partitioned by learner_id)
    RL_TRANSITION = "rl_transition"  # Unified state transitions (partitioned by learner_id)
    RL_PARAM = "rl_param"  # Unified parameters (partitioned by learner_id)
    RL_TASK_STAT = "rl_task_stat"  # Unified task statistics (partitioned by learner_id)

    # Mode transition learner — DEPRECATED: use rl_q_value/rl_transition/rl_task_stat with learner_id='mode_transition'
    RL_MODE_Q = "rl_mode_q"
    RL_MODE_HISTORY = "rl_mode_history"
    RL_MODE_TASK = "rl_mode_task"

    # Model selector learner — DEPRECATED: use rl_q_value/rl_param with learner_id='model_selector'
    RL_MODEL_Q = "rl_model_q"
    RL_MODEL_TASK = "rl_model_task"
    RL_MODEL_STATE = "rl_model_state"

    # Tool selector learner — DEPRECATED: use rl_q_value/rl_transition/rl_task_stat with learner_id='tool_selector'
    RL_TOOL_Q = "rl_tool_q"
    RL_TOOL_TASK = "rl_tool_task"
    RL_TOOL_OUTCOME = "rl_tool_outcome"

    # Cache eviction learner — DEPRECATED: use rl_q_value/rl_transition/rl_task_stat with learner_id='cache_eviction'
    RL_CACHE_Q = "rl_cache_q"
    RL_CACHE_TOOL = "rl_cache_tool"
    RL_CACHE_HISTORY = "rl_cache_history"

    # Grounding threshold learner — DEPRECATED: use rl_param/rl_transition/rl_task_stat with learner_id='grounding_threshold'
    RL_GROUNDING_PARAM = "rl_grounding_param"
    RL_GROUNDING_STAT = "rl_grounding_stat"
    RL_GROUNDING_HISTORY = "rl_grounding_history"

    # Single-table learners — DEPRECATED: use rl_task_stat/rl_param with the respective learner_id
    RL_SEMANTIC_STAT = "rl_semantic_stat"  # → learner_id='semantic_threshold'
    RL_PATIENCE_STAT = (
        "rl_patience_stat"  # → learner_id='continuation_patience' (rl_param + rl_task_stat)
    )
    RL_PROMPT_STAT = (
        "rl_prompt_stat"  # → learner_id='continuation_prompt' (rl_param + rl_task_stat)
    )
    RL_QUALITY_WEIGHT = (
        "rl_quality_weight"  # → learner_id='quality_weight' (rl_param + rl_transition)
    )
    RL_QUALITY_HISTORY = "rl_quality_history"  # → learner_id='quality_weight' rl_transition

    # Provider routing stats (smart routing performance tracker)
    RL_PROVIDER_STAT = "rl_provider_stat"  # Per-request provider telemetry (latency, success)

    # Context pruning (token optimization) — DEPRECATED: use rl_q_value/rl_task_stat with learner_id='context_pruning'
    RL_CONTEXT_PRUNING = "rl_context_pruning"

    # Cross-vertical learning
    RL_PATTERN = "rl_pattern"  # Cross-vertical patterns
    RL_PATTERN_USE = "rl_pattern_use"  # Pattern application tracking

    # Edge-classifier / decision learning (FEP-0012)
    # decision_log + decision_outcome are GLOBAL (join rl_outcome for training);
    # local_classifier_delta is a PROJECT table (per-project RL personalization).
    DECISION_LOG = "decision_log"  # Correlated decision records (training input)
    DECISION_OUTCOME = "decision_outcome"  # decision -> outcome/reward junction
    LOCAL_CLASSIFIER_DELTA = "local_classifier_delta"  # per-project RL weight overlay

    # ===========================================
    # AGENT DOMAIN (agent_)
    # ===========================================

    # Team execution
    AGENT_TEAM_CONFIG = "agent_team_config"  # Team composition Q-values
    AGENT_TEAM_RUN = "agent_team_run"  # Team execution records

    # Workflow execution
    AGENT_WORKFLOW_RUN = "agent_workflow_run"  # Workflow execution records
    AGENT_WORKFLOW_Q = (
        "agent_workflow_q"  # DEPRECATED: use rl_q_value with learner_id='workflow_execution'
    )

    # Prompt templates
    AGENT_PROMPT_STYLE = "agent_prompt_style"  # Prompt style definitions
    AGENT_PROMPT_ELEMENT = "agent_prompt_element"  # Prompt components
    AGENT_PROMPT_HISTORY = (
        "agent_prompt_history"  # DEPRECATED: use rl_transition with learner_id='prompt_template'
    )
    AGENT_PROMPT_CANDIDATE = "agent_prompt_candidate"  # GEPA-evolved prompt candidates
    AGENT_PROMPT_PARETO_INSTANCE = "agent_prompt_pareto_instance"  # GEPA v2 Pareto instances

    # Curriculum & Policy
    AGENT_CURRICULUM_STAGE = "agent_curriculum_stage"  # Learning curriculum
    AGENT_CURRICULUM_METRIC = "agent_curriculum_metric"  # Curriculum performance
    AGENT_POLICY_SNAPSHOT = "agent_policy_snapshot"  # Policy state snapshots

    # ===========================================
    # VERTICAL DOMAIN (vertical_) - Project-level tables
    # ===========================================

    # Vertical state externalization (Phase 4.3)
    VERTICAL_STATE = "vertical_state"  # Vertical context persistence
    VERTICAL_NEGOTIATION = "vertical_negotiation"  # Capability negotiation results
    VERTICAL_SESSION = "vertical_session"  # Vertical session tracking

    # ===========================================
    # GRAPH DOMAIN (graph_) - Project-level tables
    # ===========================================

    # Core graph tables (stored in project.db)
    GRAPH_NODE = "graph_node"  # Symbol nodes (functions, classes, etc.)
    GRAPH_EDGE = "graph_edge"  # References, calls, imports between nodes
    GRAPH_FILE_MTIME = "graph_file_mtime"  # File modification times for staleness

    # File-level granularity tracking (v2.0 - incremental updates)
    EMBEDDING_FILE_MAPPING = "embedding_file_mapping"  # Embedding -> file mappings

    # Module-level metrics (WS-1: graph analysis)
    GRAPH_MODULE_METRIC = "graph_module_metric"  # Module coupling/cohesion/hotspot metrics
    GRAPH_MODULE_METRIC_HISTORY = "graph_module_metric_history"  # Historical metric snapshots

    # CCG and Graph RAG (Phase 14: Graph-Based Enhancements)
    GRAPH_REQUIREMENT = "graph_requirement"  # Requirement nodes (GraphCodeAgent pattern)
    GRAPH_SUBGRAPH = "graph_subgraph"  # Cached subgraphs for multi-hop retrieval
    GRAPH_SUBGRAPH_NODE = "graph_subgraph_node"  # Junction table: subgraph -> nodes

    # ===========================================
    # UI DOMAIN (ui_)
    # ===========================================
    UI_SESSION = "ui_session"  # Interactive CLI session persistence
    UI_FAILED_CALL = "ui_failed_call"  # Failed tool call signatures

    # ===========================================
    # LEGACY TABLE MAPPINGS (for migration)
    # ===========================================
    # Maps old table names to new names for migration scripts

    @classmethod
    def get_legacy_mapping(cls) -> Dict[str, str]:
        """Get mapping from legacy table names to new names.

        This mapping is for reference only - data migration from legacy tables
        is not supported. Delete old databases and reinitialize instead.

        Returns:
            Dict mapping old_name -> new_name
        """
        return {
            # Graph (legacy -> new)
            "nodes": cls.GRAPH_NODE,
            "edges": cls.GRAPH_EDGE,
            "file_mtimes": cls.GRAPH_FILE_MTIME,
            # Mode learning (legacy -> new)
            "q_values": cls.RL_MODE_Q,
            "task_stats": cls.RL_MODE_TASK,
            "transition_history": cls.RL_MODE_HISTORY,
        }


class LearnerID:
    """Learner identifiers for partitioning unified tables.

    When using consolidated tables like rl_q_value, use these IDs
    to partition data by learner.
    """

    MODE_TRANSITION = "mode_transition"
    MODEL_SELECTOR = "model_selector"
    TOOL_SELECTOR = "tool_selector"
    CACHE_EVICTION = "cache_eviction"
    GROUNDING_THRESHOLD = "grounding_threshold"
    QUALITY_WEIGHT = "quality_weight"
    CONTINUATION_PATIENCE = "continuation_patience"
    CONTINUATION_PROMPT = "continuation_prompt"
    SEMANTIC_THRESHOLD = "semantic_threshold"
    PROMPT_TEMPLATE = "prompt_template"
    CROSS_VERTICAL = "cross_vertical"
    WORKFLOW_EXECUTION = "workflow_execution"
    TEAM_COMPOSITION = "team_composition"
    PROMPT_OPTIMIZER = "prompt_optimizer"
    PROVIDER_ROUTING = "provider_routing"

    @classmethod
    def all(cls) -> List[str]:
        """Get all learner IDs."""
        return [
            cls.MODE_TRANSITION,
            cls.MODEL_SELECTOR,
            cls.TOOL_SELECTOR,
            cls.CACHE_EVICTION,
            cls.GROUNDING_THRESHOLD,
            cls.QUALITY_WEIGHT,
            cls.CONTINUATION_PATIENCE,
            cls.CONTINUATION_PROMPT,
            cls.SEMANTIC_THRESHOLD,
            cls.PROMPT_TEMPLATE,
            cls.CROSS_VERTICAL,
            cls.WORKFLOW_EXECUTION,
            cls.TEAM_COMPOSITION,
            cls.PROMPT_OPTIMIZER,
        ]


class Schema:
    """SQL schema definitions for all tables.

    Use these to create tables with consistent structure.
    """

    # ===========================================
    # SYSTEM TABLES
    # ===========================================

    SYS_METADATA = f"""
        CREATE TABLE IF NOT EXISTS {Tables.SYS_METADATA} (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """

    SYS_SCHEMA_VERSION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.SYS_SCHEMA_VERSION} (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime('now'))
        )
    """

    # ===========================================
    # RL TABLES
    # ===========================================

    RL_LEARNER = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_LEARNER} (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            config TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        )
    """

    RL_OUTCOME = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_OUTCOME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            task_type TEXT,
            vertical TEXT DEFAULT '',
            repo_id TEXT DEFAULT NULL,
            success INTEGER,
            quality_score REAL,
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    RL_OUTCOME_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_rl_outcome_learner
            ON {Tables.RL_OUTCOME}(learner_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_rl_outcome_context
            ON {Tables.RL_OUTCOME}(provider, model, task_type);
    """

    # FEP-0012: correlated decision records — the structured mirror of
    # decisions.jsonl, carrying the correlation spine so each decision joins to
    # its outcome (rl_outcome / usage.jsonl) for reward-weighted training.
    DECISION_LOG = f"""
        CREATE TABLE IF NOT EXISTS {Tables.DECISION_LOG} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            turn_id TEXT DEFAULT '',
            trace_id TEXT DEFAULT '',
            source TEXT DEFAULT '',
            confidence REAL DEFAULT 0.0,
            model_version TEXT DEFAULT '',
            feature_spec_version TEXT DEFAULT '',
            feature_digest TEXT DEFAULT '',
            context TEXT DEFAULT '',
            result TEXT DEFAULT '',
            ts TEXT DEFAULT (datetime('now'))
        )
    """

    DECISION_LOG_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_decision_log_correlation
            ON {Tables.DECISION_LOG}(session_id, turn_id);
        CREATE INDEX IF NOT EXISTS idx_decision_log_type_ts
            ON {Tables.DECISION_LOG}(decision_type, ts);
        CREATE INDEX IF NOT EXISTS idx_decision_log_id
            ON {Tables.DECISION_LOG}(decision_id);
    """

    # FEP-0012: decision -> outcome junction. attributed_reward is the
    # per-decision credit (GAE) derived from the session outcome.
    DECISION_OUTCOME = f"""
        CREATE TABLE IF NOT EXISTS {Tables.DECISION_OUTCOME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            turn_id TEXT DEFAULT '',
            success INTEGER,
            quality_score REAL,
            attributed_reward REAL,
            credit_method TEXT DEFAULT '',
            segment_rewards TEXT DEFAULT '',
            ts TEXT DEFAULT (datetime('now'))
        )
    """

    DECISION_OUTCOME_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_decision_outcome_id
            ON {Tables.DECISION_OUTCOME}(decision_id);
        CREATE INDEX IF NOT EXISTS idx_decision_outcome_session
            ON {Tables.DECISION_OUTCOME}(session_id);
    """

    # FEP-0012: per-project RL weight overlay (personalization delta). Sparse,
    # FEP-0012 Phase 6: per-label RL weight overlay. One row per
    # (decision_type, feature_hash, label) so a multi-class head (e.g.
    # task_completion's fail/partial/pass) can carry a per-label nudge.
    # feature_spec_version pins the hasher config: a future spec bump
    # invalidates stale rows (the loader filters to the runtime spec).
    # Top-K bounded per (decision_type, label), L2-decayed. PROJECT db.
    LOCAL_CLASSIFIER_DELTA = f"""
        CREATE TABLE IF NOT EXISTS {Tables.LOCAL_CLASSIFIER_DELTA} (
            decision_type TEXT NOT NULL,
            feature_hash INTEGER NOT NULL,
            label TEXT NOT NULL,
            weight REAL DEFAULT 0.0,
            samples INTEGER DEFAULT 0,
            sum_reward REAL DEFAULT 0.0,
            feature_spec_version TEXT NOT NULL DEFAULT '',
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (decision_type, feature_hash, label, feature_spec_version)
        )
    """

    LOCAL_CLASSIFIER_DELTA_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_local_classifier_delta_type
            ON {Tables.LOCAL_CLASSIFIER_DELTA}(decision_type, feature_spec_version);
    """

    RL_Q_VALUE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_Q_VALUE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            state_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            q_value REAL DEFAULT 0.5,
            visit_count INTEGER DEFAULT 0,
            last_updated TEXT DEFAULT (datetime('now')),
            UNIQUE(learner_id, state_key, action_key)
        )
    """

    RL_Q_VALUE_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_rl_q_learner
            ON {Tables.RL_Q_VALUE}(learner_id);
    """

    RL_TRANSITION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_TRANSITION} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT,
            action TEXT,
            reward REAL,
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    RL_TRANSITION_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_rl_trans_learner
            ON {Tables.RL_TRANSITION}(learner_id, created_at);
    """

    RL_PARAM = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_PARAM} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            param_key TEXT NOT NULL,
            param_value REAL,
            value_text TEXT,
            context TEXT,
            sample_count INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.5,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(learner_id, param_key, context)
        )
    """

    RL_TASK_STAT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_TASK_STAT} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            stat_key TEXT NOT NULL,
            stat_value REAL,
            sample_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(learner_id, task_type, stat_key)
        )
    """

    RL_PROVIDER_STAT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_PROVIDER_STAT} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            task_type TEXT DEFAULT 'default',
            success INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            error_type TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    RL_PROVIDER_STAT_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_rl_pstat_provider
            ON {Tables.RL_PROVIDER_STAT}(provider, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_rl_pstat_model
            ON {Tables.RL_PROVIDER_STAT}(provider, model, created_at DESC);
    """

    RL_PATTERN = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_PATTERN} (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            pattern_name TEXT,
            avg_quality REAL,
            confidence REAL,
            source_verticals TEXT,
            recommendation TEXT,
            sample_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        )
    """

    RL_PATTERN_USE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_PATTERN_USE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id TEXT NOT NULL,
            target_vertical TEXT,
            success INTEGER,
            quality_score REAL,
            applied_at TEXT DEFAULT (datetime('now'))
        )
    """

    RL_METRIC = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_METRIC} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT,
            metric_type TEXT NOT NULL,
            metric_value REAL,
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    # ===========================================
    # AGENT TABLES
    # ===========================================

    AGENT_TEAM_CONFIG = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_TEAM_CONFIG} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key TEXT UNIQUE NOT NULL,
            formation TEXT NOT NULL,
            role_counts TEXT NOT NULL,
            task_category TEXT,
            execution_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            avg_quality REAL DEFAULT 0.5,
            avg_duration REAL DEFAULT 0,
            q_value REAL DEFAULT 0.5,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_TEAM_RUN = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_TEAM_RUN} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            task_category TEXT,
            formation TEXT,
            role_counts TEXT,
            member_count INTEGER,
            budget_used INTEGER,
            tools_used INTEGER,
            success INTEGER,
            quality_score REAL,
            duration_seconds REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_WORKFLOW_RUN = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_WORKFLOW_RUN} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_name TEXT NOT NULL,
            task_type TEXT,
            success INTEGER,
            duration_seconds REAL,
            quality_score REAL,
            vertical TEXT,
            mode TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_PROMPT_STYLE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_STYLE} (
            id TEXT PRIMARY KEY,
            style_name TEXT NOT NULL,
            description TEXT,
            template TEXT,
            success_rate REAL DEFAULT 0.5,
            usage_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_PROMPT_CANDIDATE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_CANDIDATE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section_name TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'default',
            text_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            generation INTEGER DEFAULT 0,
            parent_hash TEXT,
            completion_score REAL DEFAULT 0.0,
            token_efficiency REAL DEFAULT 0.0,
            tool_effectiveness REAL DEFAULT 0.0,
            alpha REAL DEFAULT 1.0,
            beta REAL DEFAULT 1.0,
            sample_count INTEGER DEFAULT 0,
            instance_scores TEXT DEFAULT '{{}}',
            coverage_count INTEGER DEFAULT 0,
            is_on_frontier INTEGER DEFAULT 1,
            char_length INTEGER DEFAULT 0,
            benchmark_score REAL DEFAULT 0.0,
            benchmark_runs INTEGER DEFAULT 0,
            benchmark_passed INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 0,
            strategy_name TEXT DEFAULT 'gepa',
            strategy_chain TEXT DEFAULT 'gepa',
            requires_benchmark INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(section_name, provider, text_hash)
        )
    """

    AGENT_PROMPT_PARETO_INSTANCE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_PARETO_INSTANCE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section_name TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'default',
            instance_id TEXT NOT NULL,
            best_candidate_hash TEXT,
            best_score REAL DEFAULT 0.0,
            sample_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(section_name, provider, instance_id)
        )
    """

    AGENT_PROMPT_ELEMENT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_PROMPT_ELEMENT} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style_id TEXT,
            element_type TEXT,
            content TEXT,
            position INTEGER
        )
    """

    AGENT_CURRICULUM_STAGE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_CURRICULUM_STAGE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage_name TEXT NOT NULL,
            difficulty REAL,
            prerequisites TEXT,
            completion_criteria TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_CURRICULUM_METRIC = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_CURRICULUM_METRIC} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage_id INTEGER,
            metric_name TEXT,
            metric_value REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    AGENT_POLICY_SNAPSHOT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.AGENT_POLICY_SNAPSHOT} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id TEXT,
            snapshot_name TEXT,
            policy_data TEXT,
            performance_score REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    # ===========================================
    # UI TABLES
    # ===========================================

    UI_SESSION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.UI_SESSION} (
            id TEXT PRIMARY KEY,
            name TEXT,
            provider TEXT,
            model TEXT,
            profile TEXT,
            data TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT
        )
    """

    UI_SESSION_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_ui_session_updated
            ON {Tables.UI_SESSION}(updated_at DESC);
    """

    UI_FAILED_CALL = f"""
        CREATE TABLE IF NOT EXISTS {Tables.UI_FAILED_CALL} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            args_hash TEXT NOT NULL,
            args_json TEXT,
            error_message TEXT,
            failure_count INTEGER DEFAULT 1,
            first_seen REAL,
            last_seen REAL,
            expires_at REAL,
            UNIQUE(tool_name, args_hash)
        )
    """

    UI_FAILED_CALL_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_ui_failed_lookup
            ON {Tables.UI_FAILED_CALL}(tool_name, args_hash);
        CREATE INDEX IF NOT EXISTS idx_ui_failed_expires
            ON {Tables.UI_FAILED_CALL}(expires_at);
    """

    # ===========================================
    # GRAPH TABLES (project-level)
    # ===========================================

    GRAPH_NODE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_NODE} (
            node_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            file TEXT NOT NULL,
            line INTEGER,
            end_line INTEGER,
            lang TEXT,
            signature TEXT,
            docstring TEXT,
            parent_id TEXT,
            embedding_ref TEXT,
            metadata TEXT,
            FOREIGN KEY (parent_id) REFERENCES {Tables.GRAPH_NODE}(node_id)
        )
    """

    GRAPH_EDGE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_EDGE} (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL,
            file TEXT,
            metadata TEXT,
            PRIMARY KEY (src, dst, type)
        )
    """

    GRAPH_FILE_MTIME = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_FILE_MTIME} (
            file TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            indexed_at REAL NOT NULL
        )
    """

    GRAPH_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_graph_node_type_name
            ON {Tables.GRAPH_NODE}(type, name);
        CREATE INDEX IF NOT EXISTS idx_graph_node_file
            ON {Tables.GRAPH_NODE}(file);
        CREATE INDEX IF NOT EXISTS idx_graph_node_parent
            ON {Tables.GRAPH_NODE}(parent_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edge_src_type
            ON {Tables.GRAPH_EDGE}(src, type);
        CREATE INDEX IF NOT EXISTS idx_graph_edge_dst_type
            ON {Tables.GRAPH_EDGE}(dst, type);
        CREATE INDEX IF NOT EXISTS idx_graph_file_mtime
            ON {Tables.GRAPH_FILE_MTIME}(mtime);
    """

    # Module-level metrics (WS-1)
    GRAPH_MODULE_METRIC = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_MODULE_METRIC} (
            module_path       TEXT PRIMARY KEY,
            pagerank_score    REAL DEFAULT 0.0,
            betweenness       REAL DEFAULT 0.0,
            afferent_coupling INTEGER DEFAULT 0,
            efferent_coupling INTEGER DEFAULT 0,
            instability       REAL DEFAULT 0.0,
            abstractness      REAL DEFAULT 0.0,
            distance_main_seq REAL DEFAULT 0.0,
            cohesion_lcom4    REAL DEFAULT 0.0,
            hotspot_score     REAL DEFAULT 0.0,
            symbol_count      INTEGER DEFAULT 0,
            change_frequency  INTEGER DEFAULT 0,
            tdd_priority      REAL DEFAULT 0.0,
            computed_at       TEXT DEFAULT (datetime('now'))
        )
    """

    GRAPH_MODULE_METRIC_HISTORY = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_MODULE_METRIC_HISTORY} (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            module_path   TEXT NOT NULL,
            hotspot_score REAL,
            tdd_priority  REAL,
            computed_at   TEXT DEFAULT (datetime('now'))
        )
    """

    GRAPH_MODULE_METRIC_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_gmm_hotspot
            ON {Tables.GRAPH_MODULE_METRIC}(hotspot_score DESC);
        CREATE INDEX IF NOT EXISTS idx_gmm_tdd
            ON {Tables.GRAPH_MODULE_METRIC}(tdd_priority DESC);
    """

    # Requirement nodes (GraphCodeAgent pattern)
    GRAPH_REQUIREMENT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_REQUIREMENT} (
            requirement_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            source TEXT,
            title TEXT NOT NULL,
            description TEXT,
            priority REAL DEFAULT 0.5,
            status TEXT DEFAULT 'open',
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (requirement_id) REFERENCES {Tables.GRAPH_NODE}(node_id)
        )
    """

    # Cached subgraphs for multi-hop retrieval
    GRAPH_SUBGRAPH = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_SUBGRAPH} (
            subgraph_id TEXT PRIMARY KEY,
            anchor_node_id TEXT NOT NULL,
            radius INTEGER NOT NULL,
            edge_types TEXT,
            node_count INTEGER,
            computed_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (anchor_node_id) REFERENCES {Tables.GRAPH_NODE}(node_id)
        )
    """

    # Junction table: subgraph -> nodes (many-to-many)
    GRAPH_SUBGRAPH_NODE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.GRAPH_SUBGRAPH_NODE} (
            subgraph_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            hop_distance INTEGER,
            PRIMARY KEY (subgraph_id, node_id),
            FOREIGN KEY (subgraph_id) REFERENCES {Tables.GRAPH_SUBGRAPH}(subgraph_id) ON DELETE CASCADE,
            FOREIGN KEY (node_id) REFERENCES {Tables.GRAPH_NODE}(node_id) ON DELETE CASCADE
        )
    """

    # Indexes for new CCG tables
    GRAPH_RAG_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_graph_requirement_type
            ON {Tables.GRAPH_REQUIREMENT}(type);
        CREATE INDEX IF NOT EXISTS idx_graph_requirement_status
            ON {Tables.GRAPH_REQUIREMENT}(status);
        CREATE INDEX IF NOT EXISTS idx_graph_requirement_priority
            ON {Tables.GRAPH_REQUIREMENT}(priority DESC);
        CREATE INDEX IF NOT EXISTS idx_graph_subgraph_anchor
            ON {Tables.GRAPH_SUBGRAPH}(anchor_node_id);
        CREATE INDEX IF NOT EXISTS idx_graph_subgraph_node
            ON {Tables.GRAPH_SUBGRAPH_NODE}(node_id);
        CREATE INDEX IF NOT EXISTS idx_graph_subgraph_node_distance
            ON {Tables.GRAPH_SUBGRAPH_NODE}(hop_distance);
    """

    # ===========================================
    # CONVERSATION TABLES (project-level)
    # ===========================================

    CONV_MESSAGE = """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            token_count INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            tool_name TEXT,
            tool_call_id TEXT,
            metadata TEXT,
            agent_id TEXT,
            parent_session_id TEXT,
            team_id TEXT,
            member_id TEXT,
            plan_id TEXT,
            plan_step_id TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                ON DELETE CASCADE
        )
    """

    CONV_SESSION = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            last_activity TIMESTAMP NOT NULL,
            project_path TEXT,
            provider TEXT,
            model TEXT,
            profile TEXT
        )
    """

    CONV_CONTEXT_SIZE = """
        CREATE TABLE IF NOT EXISTS context_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            token_count INTEGER,
            message_count INTEGER,
            timestamp TIMESTAMP DEFAULT (datetime('now'))
        )
    """

    CONV_CONTEXT_SUMMARY = """
        CREATE TABLE IF NOT EXISTS context_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            summary TEXT,
            timestamp TIMESTAMP DEFAULT (datetime('now'))
        )
    """

    CONV_COMPACTION_EVENT = """
        CREATE TABLE IF NOT EXISTS compaction_events (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_id TEXT,
            parent_session_id TEXT,
            team_id TEXT,
            member_id TEXT,
            plan_id TEXT,
            plan_step_id TEXT,
            strategy TEXT NOT NULL,
            messages_removed INTEGER DEFAULT 0,
            tokens_freed INTEGER DEFAULT 0,
            summary TEXT,
            created_at TIMESTAMP NOT NULL,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                ON DELETE CASCADE
        )
    """

    CONV_INDEXES = """
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON messages(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_session_agent_time
            ON messages(session_id, agent_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_messages_agent_time
            ON messages(agent_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_messages_team_time
            ON messages(team_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_messages_plan_step_time
            ON messages(plan_id, plan_step_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_compaction_events_session_agent_time
            ON compaction_events(session_id, agent_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_compaction_events_team_time
            ON compaction_events(team_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_sessions_project
            ON sessions(project_path);
        CREATE INDEX IF NOT EXISTS idx_sessions_activity
            ON sessions(last_activity DESC);
    """

    # ===========================================
    # VERTICAL STATE TABLES (project-level) - Phase 4.3
    # ===========================================

    VERTICAL_STATE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.VERTICAL_STATE} (
            id TEXT PRIMARY KEY,
            vertical_name TEXT NOT NULL,
            vertical_version TEXT,
            project_path TEXT,
            session_id TEXT,
            state_data TEXT NOT NULL,
            config_json TEXT,
            stages_json TEXT,
            middleware_json TEXT,
            safety_patterns_json TEXT,
            enabled_tools_json TEXT,
            mode_configs_json TEXT,
            negotiation_results_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """

    VERTICAL_NEGOTIATION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.VERTICAL_NEGOTIATION} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vertical_state_id TEXT NOT NULL,
            capability_name TEXT NOT NULL,
            status TEXT NOT NULL,
            agreed_version TEXT,
            supported_features TEXT,
            unsupported_features TEXT,
            missing_required_features TEXT,
            fallback_version TEXT,
            error_message TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (vertical_state_id) REFERENCES {Tables.VERTICAL_STATE}(id)
        )
    """

    VERTICAL_SESSION = f"""
        CREATE TABLE IF NOT EXISTS {Tables.VERTICAL_SESSION} (
            id TEXT PRIMARY KEY,
            vertical_name TEXT NOT NULL,
            session_type TEXT NOT NULL,
            orchestrator_id TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT 'active',
            metadata TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    VERTICAL_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_vertical_state_name
            ON {Tables.VERTICAL_STATE}(vertical_name);
        CREATE INDEX IF NOT EXISTS idx_vertical_state_session
            ON {Tables.VERTICAL_STATE}(session_id);
        CREATE INDEX IF NOT EXISTS idx_vertical_state_project
            ON {Tables.VERTICAL_STATE}(project_path);
        CREATE INDEX IF NOT EXISTS idx_vertical_state_updated
            ON {Tables.VERTICAL_STATE}(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_vertical_negotiation_state
            ON {Tables.VERTICAL_NEGOTIATION}(vertical_state_id);
        CREATE INDEX IF NOT EXISTS idx_vertical_negotiation_capability
            ON {Tables.VERTICAL_NEGOTIATION}(capability_name);
        CREATE INDEX IF NOT EXISTS idx_vertical_session_name
            ON {Tables.VERTICAL_SESSION}(vertical_name);
        CREATE INDEX IF NOT EXISTS idx_vertical_session_status
            ON {Tables.VERTICAL_SESSION}(status);
    """

    # ===========================================
    # MODE LEARNING TABLES (project-level)
    # ===========================================

    MODE_Q_VALUE = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_MODE_Q} (
            state_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            q_value REAL NOT NULL DEFAULT 0.0,
            visit_count INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL,
            PRIMARY KEY (state_key, action_key)
        )
    """

    MODE_TASK_STAT = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_MODE_TASK} (
            task_type TEXT PRIMARY KEY,
            optimal_tool_budget INTEGER DEFAULT 10,
            avg_quality_score REAL DEFAULT 0.5,
            avg_completion_rate REAL DEFAULT 0.5,
            sample_count INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        )
    """

    MODE_TRANSITION_HISTORY = f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_MODE_HISTORY} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_name TEXT NOT NULL,
            from_mode TEXT NOT NULL,
            to_mode TEXT NOT NULL,
            trigger TEXT NOT NULL,
            state_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            reward REAL,
            timestamp TEXT NOT NULL
        )
    """

    MODE_INDEXES = f"""
        CREATE INDEX IF NOT EXISTS idx_{Tables.RL_MODE_Q}_state
            ON {Tables.RL_MODE_Q}(state_key);
        CREATE INDEX IF NOT EXISTS idx_{Tables.RL_MODE_HISTORY}_profile
            ON {Tables.RL_MODE_HISTORY}(profile_name, timestamp);
    """

    # ===========================================
    # PROFILE LEARNING TABLES (project-level)
    # ===========================================

    PROFILE_INTERACTION = """
        CREATE TABLE IF NOT EXISTS interaction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_type TEXT NOT NULL,
            context TEXT,
            response_style TEXT,
            success INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """

    PROFILE_METRIC = """
        CREATE TABLE IF NOT EXISTS profile_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            context TEXT,
            sample_count INTEGER DEFAULT 1,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(metric_name, context)
        )
    """

    PROFILE_INDEXES = """
        CREATE INDEX IF NOT EXISTS idx_interaction_type
            ON interaction_history(interaction_type);
    """

    # ===========================================
    # CHANGES TABLES (project-level)
    # ===========================================

    CHANGES_GROUP = """
        CREATE TABLE IF NOT EXISTS change_groups (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp REAL,
            description TEXT,
            tool_name TEXT,
            undone INTEGER DEFAULT 0,
            data TEXT
        )
    """

    CHANGES_FILE = """
        CREATE TABLE IF NOT EXISTS file_changes (
            id TEXT PRIMARY KEY,
            group_id TEXT,
            change_type TEXT,
            file_path TEXT,
            timestamp REAL,
            tool_name TEXT,
            original_content TEXT,
            new_content TEXT,
            original_path TEXT,
            checksum_before TEXT,
            checksum_after TEXT,
            FOREIGN KEY (group_id) REFERENCES change_groups(id)
        )
    """

    CHANGES_INDEXES = """
        CREATE INDEX IF NOT EXISTS idx_file_changes_group
            ON file_changes(group_id);
        CREATE INDEX IF NOT EXISTS idx_file_changes_path
            ON file_changes(file_path);
    """

    @classmethod
    def get_all_schemas(cls) -> List[str]:
        """Get all schema definitions in creation order."""
        return [
            # System
            cls.SYS_METADATA,
            cls.SYS_SCHEMA_VERSION,
            # RL
            cls.RL_LEARNER,
            cls.RL_OUTCOME,
            cls.RL_Q_VALUE,
            cls.RL_TRANSITION,
            cls.RL_PARAM,
            cls.RL_TASK_STAT,
            cls.RL_PATTERN,
            cls.RL_PATTERN_USE,
            cls.RL_METRIC,
            # Edge-classifier decision learning (FEP-0012) — global
            cls.DECISION_LOG,
            cls.DECISION_OUTCOME,
            # Agent
            cls.AGENT_TEAM_CONFIG,
            cls.AGENT_TEAM_RUN,
            cls.AGENT_WORKFLOW_RUN,
            cls.AGENT_PROMPT_STYLE,
            cls.AGENT_PROMPT_ELEMENT,
            cls.AGENT_PROMPT_CANDIDATE,
            cls.AGENT_PROMPT_PARETO_INSTANCE,
            cls.AGENT_CURRICULUM_STAGE,
            cls.AGENT_CURRICULUM_METRIC,
            cls.AGENT_POLICY_SNAPSHOT,
            # UI
            cls.UI_SESSION,
            cls.UI_FAILED_CALL,
        ]

    @classmethod
    def get_all_indexes(cls) -> List[str]:
        """Get all index definitions."""
        return [
            cls.RL_OUTCOME_INDEXES,
            cls.RL_Q_VALUE_INDEXES,
            cls.RL_TRANSITION_INDEXES,
            cls.UI_SESSION_INDEXES,
            cls.UI_FAILED_CALL_INDEXES,
            # FEP-0012 decision-learning indexes
            cls.DECISION_LOG_INDEXES,
            cls.DECISION_OUTCOME_INDEXES,
        ]

    @classmethod
    def get_project_schemas(cls) -> List[str]:
        """Get schema definitions for project-level tables (graph, etc.).

        Note: Conversation tables (messages, sessions, context_sizes, context_summaries)
        are managed by ConversationStore, not by ProjectDatabaseManager.
        """
        return [
            # Graph
            cls.GRAPH_NODE,
            cls.GRAPH_EDGE,
            cls.GRAPH_FILE_MTIME,
            cls.GRAPH_MODULE_METRIC,
            cls.GRAPH_MODULE_METRIC_HISTORY,
            # Mode learning
            cls.MODE_Q_VALUE,
            cls.MODE_TASK_STAT,
            cls.MODE_TRANSITION_HISTORY,
            # Profile learning
            cls.PROFILE_INTERACTION,
            cls.PROFILE_METRIC,
            # Changes
            cls.CHANGES_GROUP,
            cls.CHANGES_FILE,
            # Edge-classifier per-project RL delta (FEP-0012)
            cls.LOCAL_CLASSIFIER_DELTA,
        ]

    @classmethod
    def get_project_indexes(cls) -> List[str]:
        """Get index definitions for project-level tables.

        Note: Conversation indexes are managed by ConversationStore.
        """
        return [
            cls.GRAPH_INDEXES,
            cls.GRAPH_MODULE_METRIC_INDEXES,
            cls.MODE_INDEXES,
            cls.PROFILE_INDEXES,
            cls.CHANGES_INDEXES,
            cls.LOCAL_CLASSIFIER_DELTA_INDEXES,
        ]


# Schema version for migrations
# Version 6: Database consolidation - single canonical databases
#   - Global: ~/.victor/victor.db (user-wide data)
#   - Project: ./.victor/project.db (project-specific data)
# Edges now include optional file lineage in graph_edge for direct file-aware deletes.
# Version 7: RL unified tables - add value_text column to rl_param for JSON blob storage
#   All per-learner private tables consolidated into rl_q_value/rl_transition/rl_param/rl_task_stat.
# Version 8: FEP-0012 edge-classifier decision learning
#   - decision_log + decision_outcome (global): correlated decision records + the
#     decision->outcome/reward junction, enabling reward-weighted training.
#   - local_classifier_delta (project): per-project RL weight overlay.
# Version 9: FEP-0012 Phase 6 — per-label local_classifier_delta. The v8 scalar
#   schema could not hold a multi-class head's per-label nudge (task_completion is
#   fail/partial/pass). Redefined to one row per (decision_type, feature_hash,
#   label) + a feature_spec_version guard. The table had never been written, so
#   the v9 migration DROPs+recreates it losslessly.
CURRENT_SCHEMA_VERSION = 9


def get_migration_sql(from_version: int, to_version: int) -> List[str]:
    """Get SQL statements needed to migrate between schema versions.

    Args:
        from_version: Current schema version
        to_version: Target schema version

    Returns:
        List of SQL statements to execute
    """
    migrations: Dict[int, List[str]] = {
        # Version 1 -> 2: Rename tables to new naming convention
        2: [
            # Rename system tables
            f"ALTER TABLE _db_metadata RENAME TO {Tables.SYS_METADATA}",
            # Create new RL tables
            Schema.RL_LEARNER,
            Schema.RL_Q_VALUE,
            Schema.RL_TRANSITION,
            Schema.RL_PARAM,
            Schema.RL_TASK_STAT,
            Schema.RL_METRIC,
            # Rename existing tables
            f"ALTER TABLE rl_outcomes RENAME TO {Tables.RL_OUTCOME}",
            f"ALTER TABLE cross_vertical_patterns RENAME TO {Tables.RL_PATTERN}",
            f"ALTER TABLE cross_vertical_applications RENAME TO {Tables.RL_PATTERN_USE}",
            # Rename agent tables
            f"ALTER TABLE team_composition_stats RENAME TO {Tables.AGENT_TEAM_CONFIG}",
            f"ALTER TABLE team_execution_history RENAME TO {Tables.AGENT_TEAM_RUN}",
            f"ALTER TABLE workflow_executions RENAME TO {Tables.AGENT_WORKFLOW_RUN}",
            f"ALTER TABLE prompt_template_styles RENAME TO {Tables.AGENT_PROMPT_STYLE}",
            f"ALTER TABLE prompt_template_elements RENAME TO {Tables.AGENT_PROMPT_ELEMENT}",
            f"ALTER TABLE curriculum_stages RENAME TO {Tables.AGENT_CURRICULUM_STAGE}",
            f"ALTER TABLE curriculum_metrics RENAME TO {Tables.AGENT_CURRICULUM_METRIC}",
            f"ALTER TABLE policy_checkpoints RENAME TO {Tables.AGENT_POLICY_SNAPSHOT}",
            # Rename UI tables
            f"ALTER TABLE sessions RENAME TO {Tables.UI_SESSION}",
            f"ALTER TABLE failed_signatures RENAME TO {Tables.UI_FAILED_CALL}",
        ],
        # Version 2 -> 3: Add module-level graph metrics tables
        3: [
            Schema.GRAPH_MODULE_METRIC,
            Schema.GRAPH_MODULE_METRIC_HISTORY,
            Schema.GRAPH_MODULE_METRIC_INDEXES,
        ],
        # Version 3 -> 4: Priority 4 - Add session_id column to rl_outcome for user feedback linking
        4: [
            # Add session_id column for conversation linking
            f"ALTER TABLE {Tables.RL_OUTCOME} ADD COLUMN session_id TEXT",
            # Add indexes for performance
            f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_session ON {Tables.RL_OUTCOME}(session_id, created_at)",
            f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_repo ON {Tables.RL_OUTCOME}(repo_id, created_at)",
        ],
        # Version 4 -> 5: Phase 14 - Graph-Based Enhancements (CCG + Graph RAG)
        5: [
            # Add CCG columns to graph_node
            f"ALTER TABLE {Tables.GRAPH_NODE} ADD COLUMN ast_kind TEXT",
            f"ALTER TABLE {Tables.GRAPH_NODE} ADD COLUMN scope_id TEXT",
            f"ALTER TABLE {Tables.GRAPH_NODE} ADD COLUMN statement_type TEXT",
            f"ALTER TABLE {Tables.GRAPH_NODE} ADD COLUMN requirement_id TEXT",
            f"ALTER TABLE {Tables.GRAPH_NODE} ADD COLUMN visibility TEXT",
            # Create indexes for new query patterns
            f"CREATE INDEX IF NOT EXISTS idx_graph_node_statement_type ON {Tables.GRAPH_NODE}(statement_type)",
            f"CREATE INDEX IF NOT EXISTS idx_graph_node_requirement ON {Tables.GRAPH_NODE}(requirement_id)",
            f"CREATE INDEX IF NOT EXISTS idx_graph_node_visibility ON {Tables.GRAPH_NODE}(visibility)",
            # Partial indexes for CCG edges (performance)
            f"CREATE INDEX IF NOT EXISTS idx_graph_edge_cfg ON {Tables.GRAPH_EDGE}(src, type) WHERE type LIKE 'CFG_%'",
            f"CREATE INDEX IF NOT EXISTS idx_graph_edge_cdg ON {Tables.GRAPH_EDGE}(src, type) WHERE type LIKE 'CDG_%'",
            f"CREATE INDEX IF NOT EXISTS idx_graph_edge_ddg ON {Tables.GRAPH_EDGE}(src, type) WHERE type LIKE 'DDG_%'",
            # Create new tables for requirement tracking and subgraph caching
            Schema.GRAPH_REQUIREMENT,
            Schema.GRAPH_SUBGRAPH,
            Schema.GRAPH_SUBGRAPH_NODE,
            Schema.GRAPH_RAG_INDEXES,
        ],
        # Version 5 -> 6: Database Consolidation - Single Canonical Databases
        # This migration marks the consolidation where:
        # - Global database: ~/.victor/victor.db (user-wide: settings, API keys, RL learning)
        # - Project database: ./.victor/project.db (project-specific: graph, conversations, sessions)
        # Legacy databases are migrated to their canonical locations
        6: [
            # Add consolidation metadata to track completion
            f"INSERT OR REPLACE INTO {Tables.SYS_METADATA} (key, value, updated_at) VALUES ('consolidated_version', '6', datetime('now'))",
            # Create additional indexes for performance
            f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_vertical ON {Tables.RL_OUTCOME}(vertical, created_at)",
            f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_session ON {Tables.RL_OUTCOME}(session_id, created_at)",
            # Ensure graph node FTS is up to date
            f"CREATE INDEX IF NOT EXISTS idx_graph_node_file_line ON {Tables.GRAPH_NODE}(file, line)",
        ],
        # Version 6 -> 7: RL unified tables - wire all learners to consolidated schema
        # Adds value_text TEXT column to rl_param for JSON blob storage (model_selector
        # threshold observations, prompt_template enrichment stats, etc.)
        7: [
            f"ALTER TABLE {Tables.RL_PARAM} ADD COLUMN value_text TEXT",
            f"INSERT OR REPLACE INTO {Tables.SYS_METADATA} (key, value, updated_at) VALUES ('rl_unified_schema_version', '7', datetime('now'))",
        ],
        # Version 8 (FEP-0012): edge-classifier decision learning. Idempotent
        # CREATE TABLE — decision_log/decision_outcome are global (join
        # rl_outcome for training); local_classifier_delta is per-project.
        # Indexes are created via Schema.get_all_indexes()/get_project_indexes().
        8: [
            Schema.DECISION_LOG,
            Schema.DECISION_OUTCOME,
            Schema.LOCAL_CLASSIFIER_DELTA,
        ],
        # Version 9 (FEP-0012 Phase 6): local_classifier_delta becomes per-label.
        # The table has never been written (Phase 6 is the first writer), so
        # DROP+CREATE is lossless. DROP is a no-op on the global DB (the table is
        # project-scoped). Indexes are re-created via get_project_indexes().
        9: [
            f"DROP TABLE IF EXISTS {Tables.LOCAL_CLASSIFIER_DELTA}",
            Schema.LOCAL_CLASSIFIER_DELTA,
        ],
    }

    sql_statements = []
    for version in range(from_version + 1, to_version + 1):
        if version in migrations:
            sql_statements.extend(migrations[version])

    return sql_statements


__all__ = [
    "Tables",
    "LearnerID",
    "Schema",
    "CURRENT_SCHEMA_VERSION",
    "get_migration_sql",
]
