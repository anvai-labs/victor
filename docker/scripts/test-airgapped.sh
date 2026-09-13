#!/bin/bash
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Verify the model bundled by the full Docker target without network fallback.
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python3 <<'PYTHON'
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
embedding = model.encode("Victor offline embedding smoke", convert_to_numpy=True)
assert embedding.shape == (384,), embedding.shape
assert np.isfinite(embedding).all()
print("Bundled BGE model loads and generates 384-dimensional embeddings offline.")
PYTHON
