# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 6 ship-bar guard: the parity gate must not read the per-project
delta.

``victor ml validate`` (the task_completion parity ship bar) trains/evaluates the
*universal* artifact from the global ``decision_outcome`` junction only. The
per-project ``local_classifier_delta`` is additive and must stay invisible to the
gate so shipping the delta loop cannot regress the parity decision. This is a
static source guard so it runs in every CI shard (no sklearn/data needed).
"""

from __future__ import annotations

import inspect

from victor.ml import outcome_training, parity_gate


def test_parity_gate_does_not_reference_local_delta():
    """Neither gate module reads/writes local_classifier_delta or local_delta."""
    for mod in (parity_gate, outcome_training):
        src = inspect.getsource(mod)
        assert (
            "local_classifier_delta" not in src
        ), f"{mod.__name__} references local_classifier_delta"
        assert "local_delta" not in src, f"{mod.__name__} references local_delta"
