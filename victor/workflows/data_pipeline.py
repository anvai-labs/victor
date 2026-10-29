"""
Data pipeline with robust error recovery.

The pipeline implements the following steps:
1. fetch: GET from external API with retries and circuit breaker.
2. validate: JSON schema validation; fallback parser.
3. transform: apply user-defined transformation; log errors.
4. load: batch insert into SQLite; per-record isolation and retry.
5. notify: send notification; queue if unavailable.

Checkpointing is done after each step in a simple JSON file.
Dead‑letter queue stores unrecoverable records.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from victor.config.settings import VICTOR_DIR_NAME

# Basic configuration
LOG = logging.getLogger("data_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Constants
CHECKPOINT_DIR = Path(VICTOR_DIR_NAME) / "data" / "checkpoints"
DEAD_LETTER_DIR = Path(VICTOR_DIR_NAME) / "data" / "dead_letter"
CHECKPOINT_FILE = CHECKPOINT_DIR / "pipeline_state.json"
DB_PATH = Path(VICTOR_DIR_NAME) / "data" / "pipeline.db"
NOTIF_QUEUE_FILE = Path(VICTOR_DIR_NAME) / "data" / "notification_queue.json"

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEAD_LETTER_DIR.mkdir(parents=True, exist_ok=True)
NOTIF_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)


# Circuit breaker wrapper using canonical implementation
# Replaces local implementation with victor.providers.circuit_breaker.CircuitBreaker
from victor.core.circuit_breaker import CircuitBreakerConfig
from victor.providers.circuit_breaker import CircuitBreaker as CanonicalCircuitBreaker


class CircuitBreaker:
    """Adapter wrapper around canonical CircuitBreaker for data pipeline compatibility.

    This adapter provides the same API as the previous simple CircuitBreaker
    implementation while delegating to the canonical CircuitBreaker from
    victor.providers.circuit_breaker.

    The canonical implementation provides:
    - More robust state management (CLOSED, OPEN, HALF_OPEN states)
    - Better observability with metrics and callbacks
    - Decorator and context manager support
    - Thread-safe operations with async locks
    """

    def __init__(self, max_failures: int = 5, reset_timeout: int = 60):
        # Map legacy parameters to canonical config
        config = CircuitBreakerConfig(
            failure_threshold=max_failures,
            timeout_seconds=float(reset_timeout),
        )
        self._breaker = CanonicalCircuitBreaker.from_config("data_pipeline", config)

    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker with legacy API compatibility.

        This method adapts the canonical CircuitBreaker's record_success/record_failure
        API to match the legacy call() interface that automatically tracks success/failure.
        """
        if self.is_open():
            raise Exception("Circuit breaker open")

        try:
            result = func(*args, **kwargs)
            self._breaker.record_success()
            return result
        except Exception as exc:
            self._breaker.record_failure(exc)
            LOG.warning(
                "Circuit breaker failure: %s",
                exc,
            )
            if self._breaker.is_open:
                LOG.error("Circuit breaker opened")
            raise

    def is_open(self) -> bool:
        """Check if circuit is open (failing fast).

        The canonical CircuitBreaker automatically transitions from OPEN to HALF_OPEN
        after the timeout, so this check may trigger state transitions.
        """
        return not self._breaker.can_execute()


# Pipeline implementation
class DataPipeline:
    def __init__(
        self,
        api_url: str,
        schema: Dict[str, Any],
        transform_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        self.api_url = api_url
        self.schema = schema
        self.transform_func = transform_func
        self.circuit_breaker = CircuitBreaker()
        self.state = self._load_checkpoint()
        self.db_conn = sqlite3.connect(DB_PATH)
        self._ensure_table()

    # ---------- Checkpoint helpers ----------
    def _load_checkpoint(self) -> Dict[str, Any]:
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"step": "fetch", "records_processed": 0}

    def _save_checkpoint(self, step: str, records_processed: int):
        self.state.update({"step": step, "records_processed": records_processed})
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    # ---------- Step 1: Fetch ----------
    def fetch(self) -> List[Dict[str, Any]]:
        if self.state.get("step") != "fetch":
            LOG.info("Skipping fetch, already completed")
            return []
        import requests

        backoff = [1, 2, 4]
        for attempt, delay in enumerate(backoff, start=1):
            try:
                LOG.info("Fetching data (attempt %s)", attempt)
                resp = self.circuit_breaker.call(requests.get, self.api_url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    records = data
                else:
                    records = [data]
                LOG.info("Fetched %s records", len(records))
                self._save_checkpoint("validate", len(records))
                return records
            except Exception as exc:
                LOG.warning("Fetch attempt %s failed: %s", attempt, exc)
                if attempt == len(backoff):
                    LOG.error("All fetch attempts failed, moving to dead letter")
                    self._dead_letter(records if "records" in locals() else [], "fetch")
                    return []
                time.sleep(delay)

    # ---------- Step 2: Validate ----------
    def validate(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if self.state.get("step") != "validate":
            LOG.info("Skipping validate, already completed")
            return [], []
        from jsonschema import Draft7Validator, ValidationError

        validator = Draft7Validator(self.schema)
        valid = []
        invalid = []
        for rec in records:
            try:
                validator.validate(rec)
                valid.append(rec)
            except ValidationError as e:
                LOG.warning("Record failed validation: %s", e)
                # Fallback parser: attempt to coerce types
                try:
                    coerced = self._fallback_parse(rec)
                    validator.validate(coerced)
                    valid.append(coerced)
                except Exception:
                    invalid.append(rec)
        LOG.info("Validation complete: %s valid, %s invalid", len(valid), len(invalid))
        if invalid:
            self._dead_letter(invalid, "validate")
        self._save_checkpoint("transform", len(valid))
        return valid, invalid

    def _fallback_parse(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        # Simple example: convert numeric strings to int/float
        coerced = {}
        for k, v in rec.items():
            if isinstance(v, str):
                try:
                    if "." in v:
                        coerced[k] = float(v)
                    else:
                        coerced[k] = int(v)
                except ValueError:
                    coerced[k] = v
            else:
                coerced[k] = v
        return coerced

    # ---------- Step 3: Transform ----------
    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.state.get("step") != "transform":
            LOG.info("Skipping transform, already completed")
            return []
        transformed = []
        for rec in records:
            try:
                t = self.transform_func(rec)
                transformed.append(t)
            except Exception as exc:
                LOG.error("Transform error on record %s: %s", rec.get("id"), exc)
                # Log but continue
                self._dead_letter([rec], "transform_error")
        LOG.info("Transform complete: %s records transformed", len(transformed))
        self._save_checkpoint("load", len(transformed))
        return transformed

    # ---------- Step 4: Load ----------
    def _ensure_table(self):
        # Simple table with id and payload
        self.db_conn.execute(
            "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT)"
        )
        self.db_conn.commit()

    def load(self, records: List[Dict[str, Any]]) -> Tuple[int, int]:
        if self.state.get("step") != "load":
            LOG.info("Skipping load, already completed")
            return 0, 0
        inserted = 0
        failed = 0
        batch = []
        for rec in records:
            batch.append((rec.get("id"), json.dumps(rec)))
        try:
            self.db_conn.executemany(
                "INSERT OR REPLACE INTO records (id, payload) VALUES (?, ?)", batch
            )
            self.db_conn.commit()
            inserted += len(batch)
        except sqlite3.IntegrityError as e:
            LOG.warning("Batch insert failed: %s", e)
            failed += len(batch)
            # Retry per record
            for rec in records:
                try:
                    self.db_conn.execute(
                        "INSERT OR REPLACE INTO records (id, payload) VALUES (?, ?)",
                        (rec.get("id"), json.dumps(rec)),
                    )
                    self.db_conn.commit()
                    inserted += 1
                except Exception as exc:
                    LOG.error("Per-record insert failed for %s: %s", rec.get("id"), exc)
                    failed += 1
                    self._dead_letter([rec], "load_error")
        LOG.info("Load complete: %s inserted, %s failed", inserted, failed)
        self._save_checkpoint("notify", inserted)
        return inserted, failed

    # ---------- Step 5: Notify ----------
    def notify(self, records: List[Dict[str, Any]]):
        if self.state.get("step") != "notify":
            LOG.info("Skipping notify, already completed")
            return
        for rec in records:
            try:
                self._send_notification(rec)
            except Exception as exc:
                LOG.warning("Notification failed for %s: %s", rec.get("id"), exc)
                self._queue_notification(rec)
        LOG.info("Notification step finished")
        self._save_checkpoint("complete", len(records))

    def _send_notification(self, rec: Dict[str, Any]):
        # Dummy implementation; raise if service unavailable
        import random

        if random.random() < 0.2:  # 20% chance of failure
            raise Exception("Notification service unavailable")
        LOG.info("Notified for record %s", rec.get("id"))

    def _queue_notification(self, rec: Dict[str, Any]):
        queue = []
        if NOTIF_QUEUE_FILE.exists():
            with open(NOTIF_QUEUE_FILE, "r", encoding="utf-8") as f:
                queue = json.load(f)
        queue.append(rec)
        with open(NOTIF_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2)

    # ---------- Dead letter ----------
    def _dead_letter(self, records: List[Dict[str, Any]], reason: str):
        if not records:
            return
        file = DEAD_LETTER_DIR / f"{reason}_{int(time.time())}.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        LOG.info("Dead lettered %s records for %s", len(records), reason)

    # ---------- Run pipeline ----------
    def run(self):
        # Fetch
        fetched = self.fetch()
        if not fetched:
            LOG.error("No data fetched, aborting pipeline")
            return
        # Validate
        valid, invalid = self.validate(fetched)
        if not valid:
            LOG.error("No valid records after validation, aborting pipeline")
            return
        # Transform
        transformed = self.transform(valid)
        if not transformed:
            LOG.error("No records after transformation, aborting pipeline")
            return
        # Load
        inserted, failed = self.load(transformed)
        if inserted == 0:
            LOG.error("No records inserted, aborting pipeline")
            return
        # Notify
        self.notify(transformed)
        LOG.info(
            "Pipeline completed with %s records inserted, %s failed, %s notifications queued",
            inserted,
            failed,
            NOTIF_QUEUE_FILE.stat().st_size,
        )
