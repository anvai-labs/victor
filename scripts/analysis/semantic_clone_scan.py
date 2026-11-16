"""Cross-file semantic near-duplicate scan over the indexed symbol embeddings.

Reads the LanceDB embedding store produced by `victor graph index
--embeddings`, computes cross-FILE nearest neighbours per symbol, and reports
similarity families that jscpd's exact-token matching cannot see (evolved
copies). Complements the duplication ratchet; read-only analysis.
"""

import sys

import lancedb
import numpy as np


def main(threshold: float = 0.90, top_k: int = 3) -> int:
    db = lancedb.connect(".victor/embeddings")
    table = db.open_table("embeddings")
    rows = table.search().limit(100000).to_list()
    print(f"symbols: {len(rows)}")

    ids = [r["id"] for r in rows]
    files = [r["file_path"] for r in rows]
    names = [r["symbol_name"] for r in rows]
    kinds = [r["embedding_kind"] if "embedding_kind" in r else "" for r in rows]
    mat = np.array([r["vector"] for r in rows], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9

    n = len(rows)
    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)

    # Collect cross-file candidate pairs
    pairs = {}
    for i in range(n):
        order = np.argsort(sims[i])[::-1][:top_k]
        for j in order:
            if sims[i, j] < threshold:
                continue
            if files[i] == files[j]:
                continue  # same-file near-dups are usually cohesive overloads
            key = (min(i, j), max(i, j))
            pairs[key] = float(sims[i, j])

    # Union-find into families
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j) in pairs:
        union(i, j)

    families = {}
    for i in range(n):
        families.setdefault(find(i), []).append(i)
    families = [m for m in families.values() if len(m) >= 2]

    def score(members):
        return max(pairs.get((min(a, b), max(a, b)), 0.0) for a in members for b in members if a != b)

    families.sort(key=score, reverse=True)

    print(f"cross-file pairs >= {threshold}: {len(pairs)}")
    print(f"families: {len(families)}")
    shown = 0
    for members in families:
        if shown >= 40:
            break
        s = score(members)
        member_files = {files[m] for m in members}
        if len(member_files) < 2:
            continue
        shown += 1
        print(f"\n--- family (max-sim {s:.3f}) ---")
        for m in sorted(members, key=lambda x: files[x])[:6]:
            print(f"  {files[m]}:{names[m]} ({kinds[m]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.90))
