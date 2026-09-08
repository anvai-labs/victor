# Graph Schema Migration Guide (v4 → v5)

> **Historical v4→v5 migration reference.** The schema sketches, database paths, rollback SQL,
> and `get_schema_version()` example below belong to the original migration design; the current
> store exposes no such version-query method. Do not use the historical rollback as a current
> upgrade procedure. Use the [graph quickstart](graph-quickstart.md) to rebuild an index and the
> [store API source map](graph-api-reference.md) for current behavior.

This guide helps you migrate from the v4 graph schema to v5, which adds Code Context Graph (CCG) support.

## Overview of Changes

### New Node Fields

| Field | Type | Description |
|-------|------|-------------|
| `ast_kind` | TEXT | Tree-sitter node kind |
| `scope_id` | TEXT | Hierarchical scope identifier |
| `statement_type` | TEXT | Statement category (condition, loop, etc.) |
| `requirement_id` | TEXT | Link to requirement node |
| `visibility` | TEXT | public/private/protected |

### New Edge Types

- **CFG Edges**: `CFG_SUCCESSOR`, `CFG_TRUE`, `CFG_FALSE`, `CFG_CASE`, `CFG_DEFAULT`
- **CDG Edges**: `CDG`
- **DDG Edges**: `DDG_DEF_USE`, `DDG_RAW`, `DDG_WAR`, `DDG_WAW`
- **Requirement Edges**: `SATISFIES`, `TESTS`, `DERIVES_FROM`
- **Semantic Edges**: `SEMANTIC_SIMILAR`

### New Tables

- `graph_requirement` - Requirement nodes
- `graph_subgraph` - Cached subgraphs
- `graph_subgraph_node` - Subgraph node mappings

## Automatic Migration

The migration runs automatically when you first use Victor after upgrading:

```python
from victor.storage.graph import create_graph_store
from pathlib import Path

graph_store = create_graph_store("sqlite", None, Path("."))
await graph_store.initialize()  # Migration runs here automatically
```

No manual action required. Existing data is preserved.

## Manual Migration

If you need to run the migration manually:

```bash
# Via CLI
victor graph index --force

# This will:
# 1. Detect current schema version
# 2. Run migrations to v5
# 3. Re-index with CCG enabled
```

## Verification

Check your schema version:

```python
from victor.storage.graph import create_graph_store
from pathlib import Path

graph_store = create_graph_store("sqlite", None, Path("."))
await graph_store.initialize()

schema_version = await graph_store.get_schema_version()
print(f"Schema version: {schema_version}")  # Should be 5
```

## Rollback

If you need to rollback to v4:

```bash
# Backup current database
cp ~/.victor/graph.db ~/.victor/graph.db.backup

# Revert schema (manual SQL)
sqlite3 ~/.victor/graph.db <<EOF
-- Drop new columns
ALTER TABLE graph_node DROP COLUMN ast_kind;
ALTER TABLE graph_node DROP COLUMN scope_id;
ALTER TABLE graph_node DROP COLUMN statement_type;
ALTER TABLE graph_node DROP COLUMN requirement_id;
ALTER TABLE graph_node DROP COLUMN visibility;

-- Drop new tables
DROP TABLE IF EXISTS graph_requirement;
DROP TABLE IF EXISTS graph_subgraph;
DROP TABLE IF EXISTS graph_subgraph_node;
EOF
```

## Breaking Changes

### Python API

No breaking changes in the public API. All new fields are optional and default to `None`.

### Storage Format

The SQLite database format has changed. Old databases are automatically migrated on first access.

### Edge Type Constants

Edge types are now defined in `victor.storage.graph.edge_types`:

```python
# Old way (still works for backward compatibility)
edge_type = "CALLS"

# New way (recommended)
from victor.storage.graph.edge_types import EdgeType
edge_type = EdgeType.CALLS
```

## Migration Checklist

- [ ] Backup existing graph database
- [ ] Upgrade Victor to latest version
- [ ] Run `victor graph stats` to verify migration
- [ ] Re-index with CCG enabled: `victor graph index --ccg`
- [ ] Verify CCG nodes are created
- [ ] Test graph queries work correctly

## Performance Considerations

### Indexing Time

CCG building adds approximately 20-30% to indexing time:

| Codebase Size | v4 Indexing | v5 Indexing (with CCG) |
|---------------|-------------|------------------------|
| Small (<1K files) | ~30s | ~40s |
| Medium (1K-10K) | ~5min | ~6.5min |
| Large (>10K) | ~20min | ~26min |

### Database Size

v5 schema increases database size by approximately 15-25% due to:
- Additional node metadata
- Statement-level nodes
- Additional edges (CFG, CDG, DDG)

### Query Performance

Most queries are unaffected. CCG-specific queries may be slower without caching:

```python
# Enable subgraph caching for better performance
config = RetrievalConfig(
    use_subgraph_cache=True,  # Recommended
)
```

## Compatibility

### Backward Compatibility

- **v4 databases**: Automatically migrated to v5 on first access
- **v4 code**: Works without changes (new fields are optional)
- **Edge types**: Legacy edge types (CALLS, REFERENCES, etc.) unchanged

### Forward Compatibility

- **v5 databases**: Cannot be used with v4 Victor
- **v5 code**: New features require v5 database

## Troubleshooting

### Migration Fails

```bash
# Check database integrity
sqlite3 ~/.victor/graph.db "PRAGMA integrity_check;"

# If corrupt, restore from backup
cp ~/.victor/graph.db.backup ~/.victor/graph.db

# Re-run migration
victor graph index --force
```

### Missing CCG Nodes

```bash
# Verify CCG is enabled
victor graph stats

# Re-index with CCG explicitly enabled
victor graph index --ccg --force
```

### Slow Queries After Migration

```bash
# Rebuild indexes
sqlite3 ~/.victor/graph.db "REINDEX;"

# Or simply re-index
victor graph index --force
```

## Feature Flag Control

Control which v5 features are enabled:

```python
import os
# Disable CCG building
os.environ["VICTOR_USE_CCG"] = "false"

# Disable Graph RAG
os.environ["VICTOR_USE_GRAPH_RAG"] = "false"
```

Or in settings.yaml:

```yaml
search:
  graph:
    enable_ccg: false  # Disable CCG
    enable_graph_rag: false  # Disable Graph RAG
```

## Support

For migration issues:
1. Check logs: `~/.victor/logs/`
2. Run diagnostics: `victor doctor --verbose`
3. Report issues: https://github.com/anthropics/victor/issues

## Additional Resources

- [Graph RAG Guide](graph-rag-guide.md) - Full feature documentation
- [API Reference](graph-api-reference.md) - Complete API documentation
- [Quick Start](graph-quickstart.md) - Getting started tutorial
