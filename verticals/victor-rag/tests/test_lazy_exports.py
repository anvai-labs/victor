"""Package discovery must not initialize optional storage backends."""

import subprocess
import sys
from pathlib import Path


def test_discovery_keeps_storage_libraries_lazy():
    package_root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import victor_rag
for name in ('lancedb', 'pyarrow', 'pandas', 'torch', 'sentence_transformers'):
    assert name not in sys.modules, name
assert 'DocumentStore' in dir(victor_rag)
assert 'DocumentChunker' in dir(victor_rag)
try:
    victor_rag.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export accepted')
assert victor_rag.DocumentStore.__name__ == 'DocumentStore'
assert victor_rag.DocumentChunker.__name__ == 'DocumentChunker'
assert victor_rag.DocumentStore is victor_rag.DocumentStore
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(package_root)], check=True)
