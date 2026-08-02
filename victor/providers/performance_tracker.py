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

"""
Provider Performance Tracking for Adaptive Routing.

This module provides performance tracking and learning for smart routing:
- RequestMetric: Individual request metrics
- ProviderPerformanceTracker: Tracks success rate, latency, trends

Persistence:
    Metrics are written through to the global database (~/.victor/victor.db)
    in the rl_provider_stat table so that routing decisions benefit from
    cross-session history. On initialization the tracker hydrates its
    in-memory window from the last `window_size` rows per provider.

Usage:
    from victor.providers.performance_tracker import ProviderPerformanceTracker, RequestMetric

    tracker = ProviderPerformanceTracker(window_size=100)

    # Record successful request (persisted to global DB)
    tracker.record_request(
        RequestMetric(
            provider="ollama",
            model="qwen3-coder:30b",
            success=True,
            latency_ms=1250.0,
            timestamp=datetime.now(),
        )
    )

    # Get provider score for routing decisions (uses cross-session history)
    score = tracker.get_provider_score("ollama")  # 0.0 to 1.0
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.core.database import DatabaseManager

logger = logging.getLogger(__name__)
_AUTO_DB = object()


@dataclass
class RequestMetric:
    """Individual request metric.

    Attributes:
        provider: Provider name (e.g., "ollama", "anthropic")
        model: Model identifier (e.g., "qwen3-coder:30b", "claude-3-5-sonnet-20241022")
        success: Whether the request succeeded
        latency_ms: Request latency in milliseconds
        timestamp: When the request occurred
        error_type: Type of error if request failed
        task_type: Task type hint for per-task routing analytics
    """

    provider: str
    model: str
    success: bool
    latency_ms: float
    timestamp: datetime
    error_type: Optional[str] = None
    task_type: str = "default"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "task_type": self.task_type,
        }


@dataclass(frozen=True)
class ProviderDegradationSnapshot:
    """Structured snapshot of recent provider degradation and recovery state."""

    provider: str
    total_requests: int
    success_rate: float
    recent_success_rate: float
    average_latency_ms: float
    recent_average_latency_ms: float
    latency_trend: str
    score: float
    failure_streak: int
    success_streak: int
    degraded: bool
    degradation_reasons: tuple[str, ...] = ()
    recovered_from_recent_incident: bool = False
    time_to_recover_seconds: Optional[float] = None
    recent_incident_failure_count: int = 0
    recent_error_types: Dict[str, int] = field(default_factory=dict)
    latest_model: Optional[str] = None
    last_failure_at: Optional[str] = None
    last_recovery_at: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize the snapshot into plain runtime metadata."""
        return {
            "provider": self.provider,
            "total_requests": self.total_requests,
            "success_rate": round(self.success_rate, 4),
            "recent_success_rate": round(self.recent_success_rate, 4),
            "average_latency_ms": round(self.average_latency_ms, 4),
            "recent_average_latency_ms": round(self.recent_average_latency_ms, 4),
            "latency_trend": self.latency_trend,
            "score": round(self.score, 4),
            "failure_streak": self.failure_streak,
            "success_streak": self.success_streak,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
            "recovered_from_recent_incident": self.recovered_from_recent_incident,
            "time_to_recover_seconds": (
                round(self.time_to_recover_seconds, 4)
                if self.time_to_recover_seconds is not None
                else None
            ),
            "recent_incident_failure_count": self.recent_incident_failure_count,
            "recent_error_types": dict(self.recent_error_types),
            "latest_model": self.latest_model,
            "last_failure_at": self.last_failure_at,
            "last_recovery_at": self.last_recovery_at,
        }


class ProviderPerformanceTracker:
    """Tracks provider performance for adaptive routing.

    Maintains a sliding window of recent requests for each provider and
    computes metrics for routing decisions:
    - Success rate: Percentage of successful requests
    - Average latency: Mean request latency
    - Latency trend: Improving, stable, or degrading

    Composite score = 40% success_rate + 30% (1/normalized_latency) + 30% trend_score

    Persistence:
        Writes through to rl_provider_stat in the global DB on every
        record_request() call.  On first access to a provider's metrics,
        hydrates the in-memory deque from the DB if it is empty (cold start).
        This means routing benefits from history across restarts.
    """

    def __init__(
        self,
        window_size: int = 100,
        db: Optional["DatabaseManager"] | object = _AUTO_DB,
    ):
        """Initialize performance tracker.

        Args:
            window_size: Number of recent requests to track per provider.
            db: DatabaseManager to persist metrics. When omitted, the tracker
                uses the global singleton (``DatabaseManager()``). Pass
                ``None`` to disable persistence and keep all metrics in-memory,
                which is useful for deterministic unit tests and session-local
                routing modes.
        """
        self.window_size = window_size
        self.metrics: Dict[str, Deque[RequestMetric]] = {}
        # Track which providers have been hydrated from DB this session
        self._hydrated: set[str] = set()

        if db is _AUTO_DB:
            try:
                from victor.core.database import DatabaseManager

                self._db: Optional["DatabaseManager"] = DatabaseManager()
                self._ensure_table()
            except Exception:
                logger.debug("Global DB unavailable; performance tracker will be in-memory only")
                self._db = None
        elif db is None:
            self._db = None
        else:
            self._db = db
            if self._db is not None:
                self._ensure_table()

        logger.debug(
            "ProviderPerformanceTracker initialized (window_size=%d, db=%s)",
            window_size,
            "enabled" if self._db else "disabled",
        )

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create rl_provider_stat table if it doesn't exist."""
        if self._db is None:
            return
        try:
            from victor.core.schema import Schema

            conn = self._db.get_connection()
            conn.executescript(Schema.RL_PROVIDER_STAT)
            for stmt in Schema.RL_PROVIDER_STAT_INDEXES.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()
        except Exception as exc:
            logger.warning("Failed to ensure rl_provider_stat table: %s", exc)
            self._db = None  # Degrade gracefully to in-memory

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self, metric: RequestMetric) -> None:
        """Write a metric to the global DB (fire-and-forget, non-fatal)."""
        if self._db is None:
            return
        try:
            self._db.execute(
                """
                INSERT INTO rl_provider_stat
                    (provider, model, task_type, success, latency_ms, error_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.provider.lower(),
                    metric.model,
                    metric.task_type,
                    1 if metric.success else 0,
                    metric.latency_ms,
                    metric.error_type,
                    metric.timestamp.isoformat(),
                ),
            )
            # Trim old rows for this provider beyond 10× window_size to bound table growth.
            # We keep 10× so that multiple concurrent sessions don't see an artificially small
            # pool when they each hold a separate in-memory window.
            self._db.execute(
                """
                DELETE FROM rl_provider_stat
                WHERE provider = ?
                  AND id NOT IN (
                      SELECT id FROM rl_provider_stat
                      WHERE provider = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (
                    metric.provider.lower(),
                    metric.provider.lower(),
                    self.window_size * 10,
                ),
            )
        except Exception as exc:
            logger.debug("rl_provider_stat write failed (non-fatal): %s", exc)

    def _hydrate(self, provider: str) -> None:
        """Populate the in-memory deque for a provider from the DB (once per session).

        Data quality filters applied:
        - Only loads data from last 30 days (stale data is unreliable)
        - Validates latency_ms is positive and reasonable (1ms to 60s)
        - Skips records with obviously corrupted timestamps
        """
        if self._db is None or provider in self._hydrated:
            return
        self._hydrated.add(provider)
        try:
            # Only load recent data (last 30 days) to avoid stale performance patterns
            from datetime import timedelta

            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

            rows = self._db.query(
                """
                SELECT provider, model, task_type, success, latency_ms, error_type, created_at
                FROM rl_provider_stat
                WHERE provider = ? AND created_at > ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (provider, cutoff_date, self.window_size),
            )
            if not rows:
                return
            if provider not in self.metrics:
                self.metrics[provider] = deque(maxlen=self.window_size)
            deq = self.metrics[provider]
            # Rows come back newest-first; insert oldest-first so the deque is in
            # chronological order for trend analysis.
            loaded_count = 0
            for row in reversed(rows):
                try:
                    ts = datetime.fromisoformat(row["created_at"])
                except (ValueError, TypeError):
                    continue  # Skip records with invalid timestamps
                # Validate latency is reasonable (1ms to 60 seconds)
                latency = float(row["latency_ms"])
                if latency < 1.0 or latency > 60000.0:
                    continue  # Skip obviously bad latency values
                deq.append(
                    RequestMetric(
                        provider=row["provider"],
                        model=row["model"],
                        task_type=row["task_type"] or "default",
                        success=bool(row["success"]),
                        latency_ms=latency,
                        error_type=row["error_type"],
                        timestamp=ts,
                    )
                )
                loaded_count += 1
            logger.debug(
                "Hydrated %d historical metrics for provider '%s' from DB (filtered from %d total rows)",
                loaded_count,
                provider,
                len(rows),
            )
        except Exception as exc:
            logger.debug("rl_provider_stat hydration failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(self, metric: RequestMetric) -> None:
        """Record a request metric.

        Args:
            metric: Request metric to record
        """
        provider = metric.provider.lower()

        if provider not in self.metrics:
            self.metrics[provider] = deque(maxlen=self.window_size)

        self.metrics[provider].append(metric)
        self._persist(metric)

        logger.debug(
            "Recorded metric for %s: success=%s, latency=%.0fms",
            provider,
            metric.success,
            metric.latency_ms,
        )

    def get_metrics(self, provider: str) -> List[RequestMetric]:
        """Get all metrics for a provider.

        Hydrates from the global DB on first access if the in-memory window
        is empty (cold start after a restart).

        Args:
            provider: Provider name

        Returns:
            List of request metrics
        """
        provider = provider.lower()
        self._hydrate(provider)
        return list(self.metrics.get(provider, []))

    def get_success_rate(self, provider: str) -> float:
        """Calculate success rate for a provider.

        Args:
            provider: Provider name

        Returns:
            Success rate from 0.0 to 1.0 (0.5 = unknown)
        """
        metrics = self.get_metrics(provider)

        if not metrics:
            return 0.5  # Unknown provider, neutral score

        successful = sum(1 for m in metrics if m.success)
        return successful / len(metrics)

    def get_average_latency(self, provider: str) -> float:
        """Calculate average latency for a provider.

        Args:
            provider: Provider name

        Returns:
            Average latency in milliseconds
        """
        metrics = self.get_metrics(provider)

        if not metrics:
            return 1000.0  # Unknown provider, default to 1s

        latencies = [m.latency_ms for m in metrics if m.success]

        if not latencies:
            return 1000.0

        return sum(latencies) / len(latencies)

    def get_latency_trend(self, provider: str) -> str:
        """Analyze latency trend for a provider.

        Compares recent latency to older latency to detect trends.

        Args:
            provider: Provider name

        Returns:
            Trend: "improving", "stable", or "degrading"
        """
        metrics = self.get_metrics(provider)

        if len(metrics) < 10:
            return "stable"  # Not enough data

        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        mid = len(sorted_metrics) // 2

        older = sorted_metrics[:mid]
        recent = sorted_metrics[mid:]

        older_latency = sum(m.latency_ms for m in older if m.success) / max(
            1, sum(1 for m in older if m.success)
        )
        recent_latency = sum(m.latency_ms for m in recent if m.success) / max(
            1, sum(1 for m in recent if m.success)
        )

        if recent_latency < older_latency * 0.9:
            return "improving"
        elif recent_latency > older_latency * 1.1:
            return "degrading"
        else:
            return "stable"

    def get_provider_score(self, provider: str) -> float:
        """Calculate composite provider score for routing decisions.

        Composite score = 40% success_rate + 30% latency_score + 30% trend_score

        Args:
            provider: Provider name

        Returns:
            Score from 0.0 to 1.0 (higher is better)
        """
        success_rate = self.get_success_rate(provider)
        avg_latency = self.get_average_latency(provider)
        trend = self.get_latency_trend(provider)

        # Normalize latency: 1.0 for 0ms, 0.0 for 5000ms+
        latency_score = max(0.0, 1.0 - (avg_latency / 5000.0))

        trend_scores = {"improving": 1.0, "stable": 0.7, "degrading": 0.3}
        trend_score = trend_scores.get(trend, 0.7)

        composite_score = 0.4 * success_rate + 0.3 * latency_score + 0.3 * trend_score

        logger.debug(
            "Provider %s score: %.2f (success=%.2f, latency=%.2f, trend=%.2f)",
            provider,
            composite_score,
            success_rate,
            latency_score,
            trend_score,
        )

        return composite_score

    def get_failure_streak(self, provider: str) -> int:
        """Return the current trailing failure streak for a provider."""
        metrics = self.get_metrics(provider)
        streak = 0
        for metric in reversed(metrics):
            if metric.success:
                break
            streak += 1
        return streak

    def get_success_streak(self, provider: str) -> int:
        """Return the current trailing success streak for a provider."""
        metrics = self.get_metrics(provider)
        streak = 0
        for metric in reversed(metrics):
            if not metric.success:
                break
            streak += 1
        return streak

    def get_degradation_snapshot(self, provider: str) -> ProviderDegradationSnapshot:
        """Return a structured degradation/recovery snapshot for a provider.

        The snapshot intentionally favors simple, deterministic signals:
        consecutive failures, recent success-rate erosion, and degrading
        latency trend. Recovery is only reported after a multi-failure incident
        is followed by a stable success streak.
        """
        provider = provider.lower()
        metrics = self.get_metrics(provider)
        if not metrics:
            return ProviderDegradationSnapshot(
                provider=provider,
                total_requests=0,
                success_rate=0.5,
                recent_success_rate=0.5,
                average_latency_ms=1000.0,
                recent_average_latency_ms=1000.0,
                latency_trend="stable",
                score=0.5,
                failure_streak=0,
                success_streak=0,
                degraded=False,
            )

        recent_window = metrics[-min(5, len(metrics)) :]
        success_rate = self.get_success_rate(provider)
        recent_success_rate = sum(1 for metric in recent_window if metric.success) / max(
            1, len(recent_window)
        )
        average_latency = self.get_average_latency(provider)
        recent_successful = [metric.latency_ms for metric in recent_window if metric.success]
        recent_average_latency = (
            (sum(recent_successful) / len(recent_successful))
            if recent_successful
            else average_latency
        )
        latency_trend = self.get_latency_trend(provider)
        score = self.get_provider_score(provider)
        failure_streak = self.get_failure_streak(provider)
        success_streak = self.get_success_streak(provider)

        degradation_reasons: list[str] = []
        if failure_streak >= 2:
            degradation_reasons.append("failure_streak")
        if len(metrics) >= 4 and recent_success_rate < 0.5:
            degradation_reasons.append("low_recent_success_rate")
        if len(metrics) >= 10 and latency_trend == "degrading":
            degradation_reasons.append("latency_trend")
        degraded = bool(degradation_reasons)

        recent_error_types = Counter(
            str(metric.error_type)
            for metric in recent_window
            if not metric.success and metric.error_type is not None
        )

        last_failure_at: Optional[str] = None
        last_recovery_at: Optional[str] = None
        recovered_from_recent_incident = False
        time_to_recover_seconds: Optional[float] = None
        recent_incident_failure_count = 0

        if success_streak >= 2:
            incident_end = len(metrics) - success_streak - 1
            if incident_end >= 0 and not metrics[incident_end].success:
                incident_start = incident_end
                while incident_start > 0 and not metrics[incident_start - 1].success:
                    incident_start -= 1
                recent_incident_failure_count = (incident_end - incident_start) + 1
                if recent_incident_failure_count >= 2:
                    first_failure = metrics[incident_start]
                    first_recovery = metrics[incident_end + 1]
                    last_failure = metrics[incident_end]
                    last_failure_at = last_failure.timestamp.isoformat()
                    last_recovery_at = first_recovery.timestamp.isoformat()
                    time_to_recover_seconds = max(
                        0.0,
                        (first_recovery.timestamp - first_failure.timestamp).total_seconds(),
                    )
                    recovered_from_recent_incident = not degraded

        if last_failure_at is None:
            for metric in reversed(metrics):
                if not metric.success:
                    last_failure_at = metric.timestamp.isoformat()
                    break

        latest_model = metrics[-1].model if metrics else None
        return ProviderDegradationSnapshot(
            provider=provider,
            total_requests=len(metrics),
            success_rate=success_rate,
            recent_success_rate=recent_success_rate,
            average_latency_ms=average_latency,
            recent_average_latency_ms=recent_average_latency,
            latency_trend=latency_trend,
            score=score,
            failure_streak=failure_streak,
            success_streak=success_streak,
            degraded=degraded,
            degradation_reasons=tuple(degradation_reasons),
            recovered_from_recent_incident=recovered_from_recent_incident,
            time_to_recover_seconds=time_to_recover_seconds,
            recent_incident_failure_count=recent_incident_failure_count,
            recent_error_types=dict(recent_error_types),
            latest_model=latest_model,
            last_failure_at=last_failure_at,
            last_recovery_at=last_recovery_at,
        )

    def get_all_scores(self) -> Dict[str, float]:
        """Get scores for all tracked providers.

        Returns:
            Dict mapping provider name to score
        """
        return {provider: self.get_provider_score(provider) for provider in self.metrics.keys()}

    def get_best_provider(self, candidates: List[str]) -> Optional[str]:
        """Get the best provider from a list of candidates.

        Args:
            candidates: List of provider names to consider

        Returns:
            Best provider name, or None if no candidates have data
        """
        if not candidates:
            return None

        scores = {provider: self.get_provider_score(provider) for provider in candidates}
        return max(scores.items(), key=lambda x: x[1])[0] if scores else None

    def reset_provider(self, provider: str) -> None:
        """Reset in-memory metrics for a provider (does not touch DB).

        Args:
            provider: Provider name
        """
        provider = provider.lower()
        if provider in self.metrics:
            del self.metrics[provider]
        self._hydrated.discard(provider)
        logger.debug("Reset in-memory metrics for provider %s", provider)

    def reset_all(self) -> None:
        """Reset all in-memory metrics (does not touch DB)."""
        self.metrics.clear()
        self._hydrated.clear()
        logger.debug("Reset all in-memory provider metrics")

    def get_stats(self) -> Dict[str, object]:
        """Get tracker statistics.

        Returns:
            Dict with tracker stats
        """
        provider_stats = {}
        for provider in list(self.metrics.keys()):
            provider_stats[provider] = {
                "total_requests": len(self.metrics[provider]),
                "success_rate": self.get_success_rate(provider),
                "average_latency_ms": self.get_average_latency(provider),
                "latency_trend": self.get_latency_trend(provider),
                "score": self.get_provider_score(provider),
                "failure_streak": self.get_failure_streak(provider),
                "success_streak": self.get_success_streak(provider),
                "degradation": self.get_degradation_snapshot(provider).to_dict(),
            }

        return {
            "window_size": self.window_size,
            "providers_tracked": len(self.metrics),
            "db_enabled": self._db is not None,
            "provider_stats": provider_stats,
        }
