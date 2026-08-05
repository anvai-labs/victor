# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Edge-classifier ML core (FEP-0012).

Versioned feature extraction + a shipped linear-head artifact + pure-numpy
inference. The *training* side (``trainer.py``) uses sklearn and is a DEV-only
dependency; the runtime (features + model) needs only numpy.

See FEP-0012 and ``victor/ml/features.py`` for the versioning contract.
"""

from victor.ml.features import (
    FEATURE_SPEC_VERSION,
    HASH_SPACE,
    extract_features,
)
from victor.ml.model import DecisionHead, EdgeClassifierModel

# Closed-loop mining: project an execution manifest into reward-labeled
# training rows for the classifier. See victor.evaluation.manifest.
from victor.ml.mining import mine, mine_detailed, train_from_manifest

# FEP-0012 Phase 6: reward-supervised training from the decision_outcome
# junction (the production path; mining is the offline manifest path).
from victor.ml.outcome_training import load_outcome_samples, train_from_outcomes

# FEP-0012 Phase 7: parity gate — validate a trained classifier on held-out
# outcomes before shipping (ship only at parity vs a majority-class baseline).
from victor.ml.parity_gate import (
    evaluate_on_holdout,
    ship_verdict,
    validate_outcome_training,
)

__all__ = [
    "FEATURE_SPEC_VERSION",
    "HASH_SPACE",
    "extract_features",
    "DecisionHead",
    "EdgeClassifierModel",
    "mine",
    "mine_detailed",
    "train_from_manifest",
    "load_outcome_samples",
    "train_from_outcomes",
    "evaluate_on_holdout",
    "ship_verdict",
    "validate_outcome_training",
]
