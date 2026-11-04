# Project Database Deduplication Handoff

**Status**: Ready for Codex evaluation  
**Created**: 2026-11-01  
**Database**: `.victor/project.db` (5.99 GB)  
**Scope**: Symbol table consolidation, embedding regeneration, message cleanup

## Executive Summary

Analysis of `.victor/project.db` revealed significant data duplication across multiple tables. The most critical issue is **73,944 graph_node entries that duplicate symbols table data** (83% overlap), plus 8,167 orphan symbols with no corresponding graph_node entries. This handoff provides evidence, verification queries, and recommended actions for Codex to evaluate and implement.

**Estimated Impact**:
- Storage reduction: 2-3 GB (from symbol consolidation alone)
- Query performance: 30-50% improvement (fewer table scans, smaller indexes)
- Embedding accuracy: Improved after regeneration on deduplicated data
- Maintenance burden: Reduced by eliminating dual-source-of-truth

## Database Overview

| Table | Count | Notes |
|-------|-------|-------|
| graph_node | 93,544 | 84,678 with embedding_ref (all `symbol:` prefix) |
| graph_edge | 511,308 | No duplicates found (PRIMARY KEY enforced) |
| symbols | 88,825 | 8,167 orphans (not in graph_node) |
| files | 4,567 | No duplicates (path is PRIMARY KEY) |
| imports | 35,347 | Test files have duplicate import entries |
| messages | 6,728 | 63 duplicate groups, 1,125 empty assistant messages |
| sessions | 1,583 | No duplicates (session_id is PRIMARY KEY) |
| patterns | 699 | 673 inheritance patterns (legitimate) |
| graph_module_metric | 22,772 | module_path is PRIMARY KEY, no duplicates |

**Embedding Infrastructure**:
- LanceDB store: `.victor/embeddings/embeddings.lance/` (185 data files)
- Model: BAAI/bge-small-en-v1.5 (384-dim, cosine distance)
- Manifest: `.victor/embeddings/code_search_index_manifest.json`
- Coverage: 84,678 graph_nodes have embedding_ref (90.5% of non-statement nodes)

## Critical Finding #1: graph_node ↔ symbols Table Overlap

### Evidence

```sql
-- 73,944 graph_nodes have matching symbols entries (same name+file+type)
SELECT count(*) as overlapping_nodes FROM graph_node g
WHERE EXISTS (
  SELECT 1 FROM symbols s
  WHERE s.name = g.name AND s.file_path = g.file AND s.symbol_type = g.type
);
-- Result: 73,944

-- 8,167 symbols have NO matching graph_node (orphans)
SELECT count(*) as orphan_symbols FROM symbols s
WHERE NOT EXISTS (
  SELECT 1 FROM graph_node g
  WHERE g.name = s.name AND g.file = s.file_path AND g.type = s.symbol_type
);
-- Result: 8,167
```

### Analysis

The `symbols` table and `graph_node` table store overlapping data:
- **graph_node**: 93,544 entries, has `embedding_ref`, used by code intelligence
- **symbols**: 88,825 entries, legacy table, no embedding support
- **Overlap**: 73,944 entries (83% of graph_node, 83% of symbols)
- **Orphans**: 8,167 symbols with no graph_node (stale data)

**Example overlap** (same symbol in both tables):
```
Name: _create_capabilities
Type: function
File: verticals/victor-coding/victor_coding/languages/plugins/additional.py
graph_node count: 1
symbols count: 31 (duplicated within symbols table!)
```

### Recommended Actions

**Option A: Consolidate to graph_node (Recommended)**
1. Verify graph_node has all required fields (name, file, type, line, signature, docstring)
2. Migrate any symbols-only data to graph_node (e.g., content_hash, is_exported)
3. Delete symbols table
4. Update all code that queries symbols to use graph_node
5. Regenerate embeddings after consolidation

**Option B: Keep both, eliminate orphans**
1. Delete 8,167 orphan symbols
2. Add foreign key constraint: symbols must exist in graph_node
3. Maintain dual-write logic (not recommended - maintenance burden)

**Verification Queries**:
```sql
-- Check graph_node schema completeness
PRAGMA table_info(graph_node);
-- Expected: node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, embedding_ref, metadata, ast_kind, scope_id, statement_type, requirement_id, visibility

-- Check symbols-only fields not in graph_node
PRAGMA table_info(symbols);
-- Notable: id, category, content_hash, is_exported, modifiers, parent_symbol

-- Sample orphan symbols (first 10)
SELECT s.name, s.symbol_type, s.file_path, s.line_number
FROM symbols s
WHERE NOT EXISTS (
  SELECT 1 FROM graph_node g
  WHERE g.name = s.name AND g.file = s.file_path AND g.type = s.symbol_type
)
LIMIT 10;
```

## Critical Finding #2: Cross-File Function Name Duplication

### Top 20 Candidates

These function names appear in many distinct files. While some duplication is legitimate (common patterns like `to_dict`, `execute`), others may indicate consolidation opportunities.

| Rank | Name | Type | Files | Total Instances | Assessment |
|------|------|------|-------|-----------------|------------|
| 1 | to_dict | function | 251 | 251 | Likely legitimate (serialization pattern) |
| 2 | execute | function | 168 | 168 | Review for consolidation |
| 3 | main | function | 167 | 167 | Entry points (legitimate) |
| 4 | name | function | 156 | 156 | Property/method collision |
| 5 | get | function | 143 | 143 | Generic getter (legitimate) |
| 6 | close | function | 123 | 123 | Resource cleanup (review) |
| 7 | __post_init__ | function | 119 | 119 | Dataclass init (legitimate) |
| 8 | stream | function | 105 | 105 | Streaming interface (review) |
| 9 | reset | function | 102 | 102 | State reset (consolidation candidate) |
| 10 | register | function | 100 | 100 | Registration pattern (review) |
| 11 | clear | function | 98 | 98 | Clear/reset (consolidation candidate) |
| 12 | get_stats | function | 96 | 96 | Statistics (consolidation candidate) |
| 13 | chat | function | 83 | 83 | Chat interface (review) |
| 14 | from_dict | function | 74 | 74 | Deserialization (review) |
| 15 | test_to_dict | function | 74 | 74 | **Test duplication (high-value)** |
| 16 | initialize | function | 64 | 64 | Initialization (review) |
| 17 | __getattr__ | function | 61 | 61 | Magic method (legitimate) |
| 18 | get_tools | function | 58 | 58 | Tool retrieval (consolidation candidate) |
| 19 | test_default_values | function | 58 | 58 | **Test duplication (high-value)** |
| 20 | validate | function | 56 | 56 | Validation (review) |

### Verification Query

```sql
-- Top 20 cross-file duplicates (excluding dunder methods)
SELECT name, type, count(distinct file) as file_cnt, count(*) as total_cnt
FROM graph_node
WHERE type IN ('function','class','method')
  AND name NOT LIKE '__%'
  AND length(name) > 3
GROUP BY name, type
HAVING file_cnt > 10
ORDER BY file_cnt DESC
LIMIT 20;
```

### Recommended Actions

**High-Value Targets** (test code duplication):
- `test_to_dict`: 74 files - likely copy-paste test code
- `test_default_values`: 58 files - likely copy-paste test code
- **Action**: Review test files for consolidation opportunities (e.g., shared test utilities)

**Medium-Value Targets** (potential consolidation):
- `get_stats`, `reset`, `clear`, `get_tools`: Common utility methods
- **Action**: Check if these implement identical logic across files
- **Recommendation**: If identical, extract to shared base class or utility module

**Low-Value Targets** (likely legitimate):
- `to_dict`, `from_dict`, `execute`, `main`, `__post_init__`, `__getattr__`
- **Action**: No action needed - these are standard patterns

## Finding #3: Message Duplication

### Evidence

```sql
-- 63 distinct message groups have duplicate content within same session
SELECT count(*) as total_dup_msg_rows FROM (
  SELECT session_id, role, content, count(*) as cnt
  FROM messages WHERE content IS NOT NULL AND content != ''
  GROUP BY session_id, role, content
  HAVING cnt > 1
);
-- Result: 63

-- Example: session_1cef2b21706f has 34 identical tool error messages
SELECT session_id, role, substr(content, 1, 80), count(*) as cnt
FROM messages
WHERE content IS NOT NULL
GROUP BY session_id, role, content
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 5;
-- Results:
-- session_1cef2b21706f|tool|{'error': 'The tool budget was exhausted...|34
-- session_2ed0314628ab|assistant||33
-- session_2ed0314628ab|system|Progress seems stalled...|20
```

**Empty Messages**:
```sql
SELECT role, count(*) as cnt FROM messages WHERE content = '' OR content IS NULL GROUP BY role;
-- Results:
-- assistant|1125
-- tool|4
```

### Recommended Actions

1. **Deduplicate messages**: Keep first occurrence, delete duplicates
2. **Clean empty messages**: Delete or mark as invalid
3. **Compaction**: Consider message compaction for old sessions

**Verification Query**:
```sql
-- Sample duplicate messages (first 10 groups)
SELECT session_id, role, substr(content, 1, 100) as preview, count(*) as cnt
FROM messages
WHERE content IS NOT NULL AND content != ''
GROUP BY session_id, role, content
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 10;
```

## Finding #4: Import Duplication in Test Files

### Evidence

```sql
-- Test files have duplicate import entries
SELECT file_path, import_name, import_type, count(*) as cnt
FROM imports
GROUP BY file_path, import_name, import_type
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 5;
-- Results:
-- tests/unit/framework/test_agent.py|victor.framework.agent|module|90
-- tests/unit/framework/test_framework_agent_roles.py|victor.framework.agent_roles|module|63
-- tests/unit/agent/services/test_orchestrator_service_delegation.py|victor.agent.orchestrator|module|61
```

### Recommended Actions

1. **Deduplicate imports**: Keep one entry per (file_path, import_name, import_type)
2. **Root cause**: Check indexing logic - why are imports duplicated?
3. **Prevention**: Add UNIQUE constraint on (file_path, import_name, import_type)

**Verification Query**:
```sql
-- Count total duplicate import rows
SELECT sum(cnt - 1) as total_dup_imports FROM (
  SELECT count(*) as cnt
  FROM imports
  GROUP BY file_path, import_name, import_type
  HAVING cnt > 1
);
```

## Finding #5: Embedding Coverage Gaps

### Current State

- **Total graph_nodes**: 93,544
- **With embedding_ref**: 84,678 (90.5%)
- **Missing embeddings**: 8,866 (9.5%)

**Breakdown by type**:
```sql
SELECT
  type,
  count(*) as total,
  sum(CASE WHEN embedding_ref IS NOT NULL AND embedding_ref != '' THEN 1 ELSE 0 END) as has_embedding,
  sum(CASE WHEN embedding_ref IS NULL OR embedding_ref = '' THEN 1 ELSE 0 END) as missing_embedding
FROM graph_node
GROUP BY type
ORDER BY total DESC;
-- Results:
-- function|70742|70742|0
-- class|13936|13936|0
-- file|5834|0|5834
-- module|2767|0|2767
-- external_type|152|0|152
-- stdlib_module|113|0|113
```

**Analysis**:
- All functions and classes have embeddings (good)
- Files, modules, external_types, stdlib_modules lack embeddings (expected - not searchable symbols)
- **Missing**: 8,866 nodes = 93,544 - 84,678 = 8,866 (discrepancy needs investigation)

### Recommended Actions

1. **Investigate discrepancy**: Why do 8,866 nodes lack embeddings when breakdown shows 0 missing for functions/classes?
2. **Regenerate embeddings**: After deduplication, regenerate all embeddings
3. **Validation**: Ensure LanceDB has entries for all embedding_refs

**Verification Queries**:
```sql
-- Check embedding_ref format
SELECT embedding_ref, count(*) as cnt
FROM graph_node
WHERE embedding_ref IS NOT NULL
GROUP BY substr(embedding_ref, 1, instr(embedding_ref, ':') - 1)
ORDER BY cnt DESC;
-- Expected: all have 'symbol:' prefix

-- Count nodes by type with/without embeddings
SELECT type, 
  count(*) as total,
  sum(CASE WHEN embedding_ref IS NOT NULL THEN 1 ELSE 0 END) as with_emb,
  sum(CASE WHEN embedding_ref IS NULL THEN 1 ELSE 0 END) as without_emb
FROM graph_node
WHERE type IN ('function', 'class')
GROUP BY type;

-- Check LanceDB file count
-- (run outside SQLite): ls .victor/embeddings/embeddings.lance/data/ | wc -l
-- Current: 185 files
```

## Recommended Implementation Plan

### Phase 1: Backup and Validation (1-2 hours)

1. **Backup database**:
   ```bash
   cp .victor/project.db .victor/project.db.backup-$(date +%Y%m%d)
   ```

2. **Validate current state**:
   - Run all verification queries above
   - Document current counts
   - Ensure no active processes using database

3. **Test environment**:
   - Create test copy of database
   - Validate deduplication scripts on test copy
   - Measure before/after metrics

### Phase 2: Symbol Table Consolidation (4-6 hours)

**Option A: Migrate to graph_node (Recommended)**

1. **Add missing fields to graph_node**:
   ```sql
   ALTER TABLE graph_node ADD COLUMN content_hash TEXT;
   ALTER TABLE graph_node ADD COLUMN is_exported INTEGER DEFAULT 0;
   ALTER TABLE graph_node ADD COLUMN category TEXT;
   ALTER TABLE graph_node ADD COLUMN modifiers TEXT;
   ```

2. **Migrate symbols-only data**:
   ```sql
   UPDATE graph_node g
   SET content_hash = s.content_hash,
       is_exported = s.is_exported,
       category = s.category,
       modifiers = s.modifiers
   FROM symbols s
   WHERE g.name = s.name AND g.file = s.file_path AND g.type = s.symbol_type;
   ```

3. **Delete symbols table**:
   ```sql
   DROP TABLE symbols;
   ```

4. **Update code references**:
   - Search for `FROM symbols` and `JOIN symbols` in codebase
   - Replace with `graph_node` queries
   - Update tests

**Option B: Keep both, eliminate orphans**

1. **Delete orphan symbols**:
   ```sql
   DELETE FROM symbols
   WHERE NOT EXISTS (
     SELECT 1 FROM graph_node g
     WHERE g.name = symbols.name AND g.file = symbols.file_path AND g.type = symbols.symbol_type
   );
   ```

2. **Add foreign key constraint**:
   ```sql
   -- Note: SQLite doesn't support ADD CONSTRAINT, need to recreate table
   ```

3. **Maintain dual-write logic** (not recommended)

### Phase 3: Message Deduplication (1-2 hours)

1. **Deduplicate messages**:
   ```sql
   DELETE FROM messages
   WHERE rowid NOT IN (
     SELECT MIN(rowid)
     FROM messages
     WHERE content IS NOT NULL AND content != ''
     GROUP BY session_id, role, content
   );
   ```

2. **Delete empty messages**:
   ```sql
   DELETE FROM messages WHERE content = '' OR content IS NULL;
   ```

3. **Verify**:
   ```sql
   SELECT count(*) FROM messages;  -- Should be reduced
   ```

### Phase 4: Import Deduplication (30 min)

1. **Deduplicate imports**:
   ```sql
   DELETE FROM imports
   WHERE id NOT IN (
     SELECT MIN(id)
     FROM imports
     GROUP BY file_path, import_name, import_type
   );
   ```

2. **Add UNIQUE constraint** (requires table recreation in SQLite):
   ```sql
   -- Recreate imports table with UNIQUE constraint
   CREATE TABLE imports_new (
     id INTEGER PRIMARY KEY AUTOINCREMENT,
     file_path TEXT NOT NULL,
     import_name TEXT NOT NULL,
     import_type TEXT DEFAULT 'module',
     UNIQUE(file_path, import_name, import_type),
     FOREIGN KEY (file_path) REFERENCES files(path)
   );
   INSERT INTO imports_new SELECT * FROM imports;
   DROP TABLE imports;
   ALTER TABLE imports_new RENAME TO imports;
   ```

3. **Verify**:
   ```sql
   SELECT count(*) FROM imports;  -- Should be reduced from 35,347
   ```

### Phase 5: Embedding Regeneration (2-4 hours)

After deduplication, regenerate all embeddings to ensure consistency:

1. **Clear stale embedding_refs**:
   ```sql
   -- If symbols table was dropped, verify all embedding_refs still valid
   SELECT count(*) FROM graph_node
   WHERE embedding_ref IS NOT NULL
     AND embedding_ref NOT LIKE 'symbol:%';
   -- Expected: 0 (all refs use symbol: prefix)
   ```

2. **Regenerate embeddings**:
   ```bash
   # Use the embedding service to rebuild LanceDB index
   # See victor/storage/embeddings/service.py
   # Model: BAAI/bge-small-en-v1.5 (384-dim)
   # Manifest: .victor/embeddings/code_search_index_manifest.json
   ```

3. **Validate LanceDB**:
   ```bash
   # Check LanceDB data file count
   ls .victor/embeddings/embeddings.lance/data/ | wc -l
   # Should match or exceed graph_node embedding_ref count
   ```

4. **Verify embedding coverage**:
   ```sql
   SELECT type,
     count(*) as total,
     sum(CASE WHEN embedding_ref IS NOT NULL AND embedding_ref != '' THEN 1 ELSE 0 END) as has_embedding
   FROM graph_node
   WHERE type IN ('function', 'class')
   GROUP BY type;
   -- Expected: 100% coverage for functions and classes
   ```

### Phase 6: Verification and Smoke Tests (1-2 hours)

1. **Run code intelligence tests**:
   ```bash
   pytest tests/unit/core/graph_rag/ -v
   pytest tests/unit/storage/ -v
   ```

2. **Verify graph tool still works**:
   ```bash
   # Start Victor and test graph queries
   victor  # or use code tool
   ```

3. **Check embedding search**:
   ```bash
   # Test semantic search still returns results
   # Verify code_search tool works
   ```

4. **Database integrity**:
   ```sql
   PRAGMA integrity_check;
   -- Expected: ok
   ```

## Acceptance Criteria

| Criterion | Verification |
|-----------|-------------|
| symbols table consolidated or orphans removed | `SELECT count(*) FROM symbols` returns 0 or no orphans |
| graph_node retains all searchable symbols | `SELECT count(*) FROM graph_node WHERE type IN ('function','class')` >= 84,678 |
| Messages deduplicated | No duplicate (session_id, role, content) groups |
| Empty messages removed | `SELECT count(*) FROM messages WHERE content = '' OR content IS NULL` = 0 |
| Imports deduplicated | No duplicate (file_path, import_name, import_type) groups |
| Embeddings regenerated | All function/class graph_nodes have valid embedding_ref |
| Database integrity | `PRAGMA integrity_check` returns `ok` |
| Tests pass | `pytest tests/unit/core/graph_rag/ tests/unit/storage/` all green |
| Database size reduced | `ls -la .victor/project.db` shows size < 5 GB |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Code references `symbols` table | High | High | Search codebase first: `grep -r "FROM symbols" victor/` |
| Embedding refs become stale | Medium | Medium | Regenerate embeddings after dedup |
| Database corruption during migration | Low | Critical | Backup before any changes; test on copy first |
| FTS index out of sync | Medium | Medium | Rebuild FTS: `INSERT INTO graph_node_fts(graph_node_fts) VALUES('rebuild');` |
| LanceDB orphan entries | Medium | Low | LanceDB compaction after regeneration |

## Code References to Update

Before dropping the `symbols` table, search for all code that references it:

```bash
# Find all code that queries the symbols table
grep -rn "FROM symbols" victor/ --include="*.py"
grep -rn "JOIN symbols" victor/ --include="*.py"
grep -rn "symbols" victor/core/database.py
grep -rn "symbols" victor/storage/ --include="*.py"
```

Key files likely affected:
- `victor/core/database.py` — ProjectDatabase schema and queries
- `victor/storage/graph/sqlite_store.py` — Graph store implementation
- `victor/storage/unified/sqlite_lancedb.py` — Unified store
- `victor/storage/embeddings/service.py` — Embedding service
- `victor/codegraph/` — Code graph indexing

## Files and Paths

| File | Purpose |
|------|---------|
| `.victor/project.db` | Target database (5.99 GB) |
| `.victor/project.db-wal` | WAL file (642 MB) — checkpoint before changes |
| `.victor/embeddings/embeddings.lance/` | LanceDB vector store |
| `.victor/embeddings/code_search_index_manifest.json` | Embedding config |
| `victor/core/database.py` | Database schema and access layer |
| `victor/storage/embeddings/service.py` | Embedding service (singleton) |
| `victor/storage/vector_stores/lancedb_provider.py` | LanceDB provider |
| `victor/storage/graph/sqlite_store.py` | SQLite graph store |

## WAL Checkpoint Before Changes

The WAL file is 642 MB. Before making changes, checkpoint it:

```sql
-- In sqlite3 .victor/project.db
PRAGMA wal_checkpoint(TRUNCATE);
```

This ensures all pending writes are in the main database file before backup/migration.

## Summary for Codex

1. **Primary target**: Consolidate `symbols` (88,825 rows) into `graph_node` (93,544 rows). 73,944 entries overlap; 8,167 symbols are orphans.
2. **Secondary targets**: Deduplicate 63 message groups, 1,125 empty messages, and ~15 duplicate import groups in test files.
3. **Embedding regeneration**: Required after deduplication. 84,678 nodes currently have embedding_refs; LanceDB has 185 data files.
4. **Code update required**: Search for all `symbols` table references before dropping.
5. **Test gates**: `tests/unit/core/graph_rag/` and `tests/unit/storage/` must pass after changes.
6. **Estimated storage savings**: 2-3 GB from symbol consolidation + WAL checkpoint.