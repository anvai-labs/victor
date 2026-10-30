# Structural clone detection

`victor graph similar --mode structural` finds functions whose **logic** is
almost identical even when variable names, constants or a few statements
differ. It complements the jscpd duplication ratchet (exact token copies)
and `--mode semantic` (embedding similarity of names, signatures and
docstrings, which pairs a Protocol with its implementation and misses copies
whose names drifted).

Implementation: `victor/core/graph_rag/structural_clones.py`.

## How it works

1. **Normalise each function body** (Python `ast`, docstring dropped, pre-order walk):
   - Locals and parameters become placeholders, numbered in order of first use.
   - Constants become their type.
   - Every other name or attribute keeps its **resolved SCIP symbol**, read from `.victor/precise/scip-python.scip`. So `registry.get` and `cache.get` stay distinct even though both are spelled `.get`.
   - `self.<member>` drops its class prefix, so the same method copied into another class still matches.
2. **Tiers**:
   - `exact`: identical token streams in different files. Copies with renamed variables or changed constants. **Merge candidate.**
   - `template`: identical once resolved callees are erased too, while the callees differ. **Kernel with an injected callable.**
   - `near`: MinHash/LSH over 5-token shingles, confirmed with exact Jaccard ≥ 0.8 (`--near-threshold`). **Edited copies.** Placeholders are unnumbered inside shingles: one inserted statement would otherwise renumber every later local and break every downstream shingle.
3. **Suppression:** a family whose members are *all* sanctioned is listed as suppressed with the reason, never dropped:
   - Protocol methods;
   - `@abstractmethod`;
   - literal-dense bodies (≥ 40% constants: data tables).
4. **SCIP freshness:**
   - **Per token:** a symbol is used only when its range spans exactly the name at that position. A shifted or stale range misses and falls back to the name.
   - **Per document:** more than 20% non-identifier ranges marks the document stale, and it is reported. Below that, mismatches are SCIP quirks (f-string offsets, `import x as y` spans).
   - **No dump:** name-only resolution, announced in the output.

## Feasibility experiments (2026-10-29, read-only, real repo)

| Approach | Result | Decision |
|---|---|---|
| SCIP-normalised hash | 15,428 functions in ~17 s; 70 exact + 85 template cross-file families. SCIP split 23 AST-only families whose copies call different functions; those became the `template` tier. | Shipped |
| MinHash/LSH near tier | 68 families at J ≥ 0.8 (125 at 0.7, noisier). numpy only. | Shipped, 0.8 default |
| Body embeddings (bge-small, GPU) | 19K bodies in 179 s (106/s); finds behaviourally similar code but noisier (same-contract providers, data builders). | Phase 3: model bake-off before adoption |
| Dependence-graph (CCG) isomorphism | Data exists (2.07M DDG edges). Little marginal value over the near tier on the misses below. | Deferred |

## Recall against past dedup PRs

**Labels:** functions (≥ 40 tokens) that a dedup PR deleted. **Hit:** the function was in a cross-file family on the PR's parent commit. AST-only, since there is no SCIP dump for historical commits.

| PRs | Removed functions flagged |
|---|---|
| #1554 #1555 #1556 #1561 #1564 #1565 #1567 #1571 | **39/56 = 70%** at J ≥ 0.8 (71% at 0.7, 75% at 0.6) |

The misses are:
- **diverged copies** whose logic genuinely differs: the per-vertical persona trait code that #1556 replaced with a table-driven kernel, and `TeamMemberSpec` copies with added fields;
- **same-file duplication**: the #1565 keyword tables. The scan is cross-file by default; `--same-file` adds same-file exact families.

Lowering the threshold buys little recall for much more noise, so 0.8 stays the default.

## Running it

```bash
victor graph index                      # builds/refreshes the SCIP dump (optional)
victor graph similar --mode structural  # exact / template / near families
victor graph similar --mode structural --scope victor/agent/ --json
```
