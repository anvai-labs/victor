# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""L2: within-task caching of the learning-trace merge (perf, flag-gated, default-off)."""

from __future__ import annotations

from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner


def _learner(ttl: float) -> PromptOptimizerLearner:
    # Bypass __init__ (DB) — exercise only _collect_learning_traces + its cache.
    lr = PromptOptimizerLearner.__new__(PromptOptimizerLearner)
    lr._use_pareto = False
    lr._traces_cache_ttl_cached = ttl  # bypass the settings read
    lr._traces_cache = None
    return lr


def _wire(monkeypatch, lr, counter):
    def _collect(limit):
        counter["n"] += 1
        return [f"trace-{counter['n']}"]

    monkeypatch.setattr(lr, "_collect_traces", _collect)
    monkeypatch.setattr(lr, "_collect_traces_from_conversations", lambda limit: [])
    monkeypatch.setattr(lr, "_merge_traces", lambda a, b: list(a))


def test_traces_memoized_when_ttl_positive(monkeypatch):
    lr = _learner(60.0)
    counter = {"n": 0}
    _wire(monkeypatch, lr, counter)

    r1 = lr._collect_learning_traces()
    r2 = lr._collect_learning_traces()

    assert counter["n"] == 1, "second call should be served from the cache"
    assert r1 is r2


def test_traces_not_cached_when_ttl_zero(monkeypatch):
    lr = _learner(0.0)  # default — caching disabled
    counter = {"n": 0}
    _wire(monkeypatch, lr, counter)

    lr._collect_learning_traces()
    lr._collect_learning_traces()

    assert counter["n"] == 2, "with TTL=0 collection runs every call (current behavior)"


def test_cache_keyed_by_limit(monkeypatch):
    lr = _learner(60.0)
    counter = {"n": 0}
    _wire(monkeypatch, lr, counter)

    lr._collect_learning_traces(limit=50)
    lr._collect_learning_traces(limit=10)  # different key -> recompute

    assert counter["n"] == 2
