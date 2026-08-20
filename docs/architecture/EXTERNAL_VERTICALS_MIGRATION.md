# External Verticals Migration Plan

**Date**: 2026-03-01
**Status**: Superseded by ADR-007 on 2026-07-02
**Historical goal**: Consolidate external vertical packages into main codebase to reduce duplication

> **Historical update (2026-03-15)**: This document previously described external standalone packages as preferred while contrib shims emitted `DeprecationWarning`.
>
> **Current policy (2026-07-02)**: [ADR-007](adr/007-vertical-distribution-and-sdk-boundary.md) is authoritative. The first-party Coding, DevOps, RAG, DataAnalysis, and Research implementations live under `verticals/` in this repository. Do not use this historical migration plan to choose a source of truth or introduce another peer implementation.

---

## Executive Summary

Victor currently has **5 external vertical packages** totaling **73,498 LOC** across separate repositories. This migration will consolidate them into the main `victor-ai` codebase under `victor/verticals/contrib/`, providing:

- **~5,000 LOC reduction** by eliminating duplicated generic capabilities
- **Single repository** for easier maintenance and development
- **Better test coverage** across all verticals
- **Simplified deployment** (single pip install)

---

## Current State

### External Verticals

| Vertical | Repository | LOC | Files | PyPI Package |
|----------|-----------|-----|-------|--------------|
| **coding** | victor-coding | 41,771 | 112 | `pip install victor-coding` |
| **rag** | victor-rag | 11,428 | 35 | `pip install victor-coding` |
| **devops** | victor-devops | 7,256 | 20 | `pip install victor-devops` |
| **dataanalysis** | victor-dataanalysis | 7,255 | 20 | `pip install victor-dataanalysis` |
| **research** | victor-research | 5,788 | 18 | `pip install victor-research` |
| **Total** | **5 repos** | **73,498** | **205** | 5 packages |

### Integration Method

External verticals integrate via **entry points**:

```python
# victor-coding/pyproject.toml
[project.entry-points."victor.verticals"]
coding = "victor_coding:CodingVertical"

# victor-rag/pyproject.toml
[project.entry-points."victor.verticals"]
rag = "victor_rag:RAGVertical"
```

---

## Migration Strategy

### Phase 1: Analysis and Planning (Week 1)

**Goal**: Identify generic capabilities and create migration plan

#### 1.1 Identify Generic Capabilities

Scan each external vertical for generic capabilities that should be promoted:

| Capability | Present In | Should Promote? | Target Location |
|------------|------------|-----------------|-----------------|
| Stage Builder | All verticals | ✅ Yes | `victor/framework/capabilities/stages.py` |
| Grounding Rules | coding, devops | ✅ Yes | `victor/framework/capabilities/grounding_rules.py` |
| Validation | All verticals | ✅ Yes | `victor/framework/capabilities/validation.py` |
| Safety Rules | All verticals | ✅ Yes | `victor/framework/capabilities/safety_rules.py` |
| Task Hints | All verticals | ✅ Yes | `victor/framework/capabilities/task_hints.py` |
| Source Verification | rag, research | ✅ Yes | `victor/framework/capabilities/source_verification.py` |
| LSP Completion | coding | ❌ No | Keep as domain-specific |
| RAG Ingestion | rag | ❌ No | Keep as domain-specific |
| Docker Operations | devops | ❌ No | Keep as domain-specific |

#### 1.2 Estimate LOC Savings

| Category | Current LOC | After Migration | Savings |
|----------|-------------|-----------------|---------|
| Duplicated Boilerplate | ~3,000 | ~500 | **2,500** |
| Generic Capabilities | ~2,000 | ~0 (promoted) | **2,000** |
| Shared Testing | ~500 | ~0 (shared) | **500** |
| **Total** | **5,500** | **500** | **~5,000** |

---

### Phase 2: Create Contrib Structure (Week 1)

**Goal**: Set up directory structure in main codebase

#### 2.1 Directory Layout

```
victor/
├── verticals/                    # NEW: All verticals consolidated
│   ├── __init__.py
│   ├── contrib/                  # Migrated external verticals
│   │   ├── __init__.py
│   │   ├── coding/               # From victor-coding package
│   │   │   ├── __init__.py
│   │   │   ├── assistant.py
│   │   │   ├── capabilities.py
│   │   │   ├── workflows/
│   │   │   ├── completion/
│   │   │   └── testgen/
│   │   ├── rag/                  # From victor-rag package
│   │   │   ├── __init__.py
│   │   │   ├── assistant.py
│   │   │   ├── capabilities.py
│   │   │   ├── tools/
│   │   │   └── chunker.py
│   │   ├── devops/               # From victor-devops package
│   │   │   ├── __init__.py
│   │   │   ├── assistant.py
│   │   │   ├── capabilities.py
│   │   │   └── teams/
│   │   ├── dataanalysis/         # From victor-dataanalysis package
│   │   │   ├── __init__.py
│   │   │   ├── assistant.py
│   │   │   └── capabilities.py
│   │   └── research/             # From victor-research package
│   │       ├── __init__.py
│   │       ├── assistant.py
│   │       └── capabilities.py
│   └── shared/                   # NEW: Shared vertical utilities
│       ├── __init__.py
│       ├── capabilities_base.py  # Base classes for capabilities
│       └── prompts_base.py       # Base prompt templates
```

#### 2.2 Implementation Steps

```bash
# 1. Create directory structure
mkdir -p victor/verticals/contrib/{coding,rag,devops,dataanalysis,research}
mkdir -p victor/verticals/shared

# 2. Create __init__.py files
touch victor/verticals/__init__.py
touch victor/verticals/contrib/__init__.py
touch victor/verticals/shared/__init__.py
```

---

### Phase 3: Migrate Verticals (Week 2)

**Goal**: Move external vertical code into main codebase

#### 3.1 Migration Process (Per Vertical)

For each vertical (coding, rag, devops, dataanalysis, research):

1. **Copy Code**
   ```bash
   cp -r ../victor-{name}/victor_{name}/* victor/verticals/contrib/{name}/
   ```

2. **Update Imports**
   - Replace `from victor_{name}` with `from victor.verticals.contrib.{name}`
   - Update relative imports
   - Run automated import fixer

3. **Promote Generic Capabilities**
   - Extract generic patterns to `victor/framework/capabilities/`
   - Replace with framework imports
   - Update capability metadata

4. **Update Entry Points**
   - Add to main `pyproject.toml`
   - Mark external packages as deprecated

5. **Migrate Tests**
   - Copy tests to `tests/unit/verticals/contrib/{name}/`
   - Update test imports
   - Ensure all tests pass

#### 3.2 Example: Coding Vertical Migration

**Before** (victor-coding package):
```python
# victor_coding/capabilities.py
from victor.framework.capabilities import BaseCapabilityProvider

class CodingCapabilityProvider(BaseCapabilityProvider):
    def get_capabilities(self):
        return [
            # Generic stage builder (DUPLICATED)
            self._build_stages(),
            # Domain-specific
            self._build_code_completion(),
        ]
```

**After** (migrated):
```python
# victor/verticals/contrib/coding/capabilities.py
from victor.framework.capabilities import (
    BaseCapabilityProvider,
    StageBuilderCapability,  # Promoted to framework
)

class CodingCapabilityProvider(BaseCapabilityProvider):
    def get_capabilities(self):
        return [
            StageBuilderCapability(),  # From framework
            self._build_code_completion(),  # Domain-specific
        ]
```

---

### Phase 4: Update Entry Points (Week 2)

**Goal**: Update vertical discovery to use contrib location

#### 4.1 Update Main pyproject.toml

```toml
[project.entry-points."victor.verticals"]
# Built-in verticals (minimal, for bootstrapping)
benchmark = "victor.benchmark:BenchmarkVertical"
classification = "victor.classification:ClassificationVertical"
iac = "victor.iac:IaCVertical"
security = "victor.security:SecurityVertical"

# Contrib verticals (migrated from external packages)
coding = "victor.verticals.contrib.coding:CodingVertical"
rag = "victor.verticals.contrib.rag:RAGVertical"
devops = "victor.verticals.contrib.devops:DevOpsVertical"
dataanalysis = "victor.verticals.contrib.dataanalysis:DataAnalysisVertical"
research = "victor.verticals.contrib.research:ResearchVertical"
```

#### 4.2 Deprecate External Packages

Add deprecation notice to external packages:

```python
# victor-coding/victor_coding/__init__.py
import warnings

warnings.warn(
    "victor-coding package is deprecated. "
    "Install 'victor-ai[full]' for all verticals. "
    "See: https://github.com/anvai-labs/victor/migration-guide",
    DeprecationWarning,
    stacklevel=2
)
```

---

### Phase 5: Testing and Validation (Week 3)

**Goal**: Ensure all verticals work correctly after migration

#### 5.1 Test Coverage

| Test Suite | Before | After | Coverage |
|------------|--------|-------|----------|
| Coding Vertical | 50 tests | 50 tests | ✅ 100% |
| RAG Vertical | 30 tests | 30 tests | ✅ 100% |
| DevOps Vertical | 25 tests | 25 tests | ✅ 100% |
| DataAnalysis Vertical | 20 tests | 20 tests | ✅ 100% |
| Research Vertical | 15 tests | 15 tests | ✅ 100% |
| Integration Tests | 0 tests | 20 tests | ✅ NEW |

#### 5.2 Validation Checklist

- [ ] All imports resolve correctly
- [ ] All verticals load via entry points
- [ ] All vertical capabilities register
- [ ] All workflows compile and execute
- [ ] All tools are discoverable
- [ ] Integration tests pass
- [ ] Performance benchmarks unchanged
- [ ] Documentation updated

---

### Phase 6: Deprecation and Cleanup (Week 4)

**Goal**: Deprecate external packages and update documentation

#### 6.1 External Package Deprecation

For each external package (victor-coding, victor-rag, etc.):

1. **Add Deprecation Notice**
   ```python
   # In __init__.py
   warnings.warn(
       "This package is deprecated. "
       "The vertical is now included in victor-ai. "
       "pip install victor-ai[full]",
       DeprecationWarning,
       stacklevel=2
   )
   ```

2. **Release Final Version**
   - Version `0.1.0.deprecated`
   - Migration notice in README
   - Link to migration guide

3. **Archive Repositories**
   - Mark as read-only
   - Add migration notice
   - Link to main repo

#### 6.2 Documentation Updates

Update `docs/verticals.md`:
```markdown
## Verticals

Victor includes 9 verticals:

### Built-in Verticals
- Benchmark
- Classification
- IaC
- Security

### Contrib Verticals
- **Coding**: Full software development assistant
- **RAG**: Retrieval-augmented generation
- **DevOps**: Infrastructure and operations
- **DataAnalysis**: Data science and analytics
- **Research**: Research and citation management

All verticals are included in `victor-ai[full]` or can be installed individually.
```

---

## Risk Assessment

### High Risk

1. **Import Breaking Changes**
   - **Risk**: External code imports from old package paths
   - **Mitigation**: Provide shim/compatibility layer
   - **Contingency**: Keep external packages as shims for 2 versions

2. **Entry Point Conflicts**
   - **Risk**: Both old and new entry points registered
   - **Mitigation**: Deprecate old entry points explicitly
   - **Contingency**: Use version pinning to prevent conflicts

### Medium Risk

1. **Test Failures**
   - **Risk**: Tests fail due to import changes
   - **Mitigation**: Comprehensive test suite before migration
   - **Contingency**: Rollback and fix issues incrementally

2. **Performance Regression**
   - **Risk**: Slower vertical loading due to monorepo
   - **Mitigation**: Benchmark before/after, optimize imports
   - **Contingency**: Lazy loading improvements

### Low Risk

1. **Documentation Gaps**
   - **Risk**: Users confused about new package structure
   - **Mitigation**: Clear migration guide
   - **Contingency**: Support during transition period

---

## Rollback Plan

Each phase can be independently rolled back:

- **Phase 1-2**: Delete created directories (< 1 day)
- **Phase 3**: Revert migration commits, keep external packages (1 day)
- **Phase 4**: Revert entry point changes (< 1 day)
- **Phase 5**: Keep tests, skip integration tests (< 1 day)
- **Phase 6**: Undeprecate packages (< 1 day)

**Total Rollback Time**: < 3 days for any phase

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| LOC Reduction | ~5,000 LOC | `cloc victor/verticals/contrib/` |
| Test Coverage | >90% | `pytest tests/unit/verticals/` |
| Import Success Rate | 100% | Test all import paths |
| Vertical Load Time | <100ms | Benchmark vertical loading |
| Migration Completion | 5/5 verticals | All verticals in contrib/ |

---

## Timeline

**Total Duration**: 4 weeks

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Analysis + Structure | Migration plan, contrib/ directory |
| 2 | Migration + Entry Points | All verticals migrated |
| 3 | Testing + Validation | All tests passing |
| 4 | Deprecation + Documentation | External packages deprecated |

---

## Next Steps

1. **Review and approve** this migration plan
2. **Create contrib/** directory structure
3. **Begin migration** with coding vertical (largest, most complex)
4. **Test thoroughly** after each vertical migration
5. **Update documentation** and deprecate external packages

---

## References

- External Repositories: `../victor-{coding,rag,devops,dataanalysis,research}/`
- Architecture Analysis: `docs/architecture/ARCHITECTURE_ANALYSIS.md`
- Vertical Extension System: `victor/core/verticals/extension_loader.py`
- Entry Point System: `pyproject.toml [project.entry-points."victor.verticals"]`
