# Performance Regression Tests

This directory contains performance regression tests for the Victor AI framework, specifically focused on tool registration and query operations.

## Purpose

Performance regression tests ensure that code changes don't introduce performance degradation. They:

- **Establish baselines**: Record performance metrics for key operations
- **Detect regressions**: Alert when performance degrades beyond thresholds
- **Track improvements**: Highlight performance optimizations
- **Guide optimization**: Identify bottlenecks and hotspots

## Running Performance Tests

### Quick Start

```bash
# Install dependencies
pip install -e ".[dev,benchmark]"

# Run all performance tests
pytest tests/performance/ --benchmark-only -v

# Run specific test groups
pytest tests/performance/ -k "registration" --benchmark-only
pytest tests/performance/ -k "batch" --benchmark-only
pytest tests/performance/ -k "queries" --benchmark-only
```

### Advanced Usage

```bash
# Generate detailed HTML report
pytest tests/performance/ --benchmark-only --benchmark-html=benchmark.html

# Save baseline for comparison
pytest tests/performance/ --benchmark-only --benchmark-autosave

# Compare against saved baseline
pytest tests/performance/ --benchmark-only --benchmark-compare

# Run with performance regression gates
pytest tests/performance/ --benchmark-only --benchmark-min-rounds=5
```

## Test Categories

### Registration Performance

Tests the performance of tool registration operations:

- `test_register_10_items` - Baseline small-scale registration
- `test_register_100_items` - Medium-scale registration
- `test_register_1000_items` - Large-scale registration

**Performance Targets:**
- 10 items: < 0.5ms
- 100 items: < 5ms
- 1000 items: < 50ms

### Batch Registration Performance

Tests the performance of batch registration with single cache invalidation:

- `test_batch_register_100_items` - Batch registration with context manager
- `test_batch_register_1000_items` - Large batch registration
- `test_batch_registration_api_100` - BatchRegistrar API
- `test_batch_registration_api_1000` - BatchRegistrar with 1000 items

**Performance Targets:**
- 100 items: < 2ms (2.5× faster than individual registration)
- 1000 items: < 20ms (2.5× faster than individual registration)

### Query Performance

Tests the performance of registry query operations:

- `test_get_by_name` - O(1) name lookup
- `test_list_all` - List all tools
- `test_get_schemas` - Schema generation with caching

### Cache Performance

Tests the effectiveness of caching optimizations:

- `test_uncached_feature_flag_checks` - Baseline feature flag checks
- `test_cached_feature_flag_checks` - Cached feature flag checks
- `test_uncached_queries` - Baseline query operations
- `test_cached_queries` - Cached query operations

## Performance Targets

Based on profiling results (`docs/performance/registration_profiling_results.md`):

| Scale | Target | Optimization |
|-------|--------|--------------|
| 10 items | < 0.5ms | Baseline |
| 100 items | < 5ms | ✅ Met |
| 1000 items | < 50ms | ✅ Met |
| 10000 items | < 500ms | Requires batch API |

### Optimization Impact

| Optimization | Expected Gain | Status |
|--------------|---------------|--------|
| Feature flag caching | 1.5× | ✅ Implemented |
| Batch cache invalidation | 2× | ✅ Implemented |
| Batch registration API | 3-5× | ✅ Implemented |
| Query result caching | 2× | ✅ Implemented |
| Async registration | 5-10× | ⏳ Pending |

## CI Integration

Performance tests run automatically in CI:

- **On push to main/develop**: Full performance suite
- **On pull requests**: Compare to baseline, alert on regression
- **Daily schedule**: Track performance trends over time
- **Manual trigger**: On-demand performance analysis

### Regression Thresholds

- **Alert threshold**: 150% of baseline (50% slower)
- **Regression threshold**: 120% of baseline (20% slower)
- **Improvement threshold**: 90% of baseline (10% faster)

### CI Gates

Performance tests use `fail-on-alert: true` in CI, meaning:

- PRs with significant regressions will fail checks
- Requires explicit override to merge
- Maintains performance quality bar

## Benchmark Data

Benchmark data is stored in `.benchmarks/` directory:

```
.benchmarks/
├── 0001_test_registration_performance.json
├── 0002_test_batch_performance.json
└── ...
```

Each file contains detailed statistics:

- **Mean**: Average execution time
- **StdDev**: Standard deviation (variability)
- **Min/Max**: Fastest/slowest runs
- **Ops**: Operations per second

## Interpreting Results

### Benchmark Output

```
---------------------------------------------------------------------------------
Name (time in ms)           Min       Max      Mean    StdDev    Median     Rounds
---------------------------------------------------------------------------------
test_register_10_items      0.0401    0.0523   0.0432   0.0031    0.0425       100
test_register_100_items     0.4012    0.5234   0.4321   0.0312    0.4251       100
test_register_1000_items    4.0123    5.2341   4.3210   0.3123    4.2510       100
---------------------------------------------------------------------------------
```

### Key Metrics

- **Mean**: Typical performance (use for comparisons)
- **StdDev**: Consistency (lower = more stable)
- **Min**: Best-case performance
- **Max**: Worst-case performance

### Performance Regression Example

```
⚠️  PERFORMANCE REGRESSIONS DETECTED
================================================================
  test_register_100_items: +35.2% slower
  test_batch_register_1000_items: +28.7% slower
================================================================
```

This indicates:
- `test_register_100_items` is 35% slower than baseline
- `test_batch_register_1000_items` is 29% slower than baseline
- Both exceed the 20% regression threshold

## Troubleshooting

### Tests Fail on CI but Pass Locally

**Cause**: Environment differences (CPU, load, etc.)

**Solution**:
- Increase regression threshold for that test
- Use `--benchmark-min-rounds` for more consistent results
- Check for resource contention on CI runner

### Highly Variable Results

**Cause**: System load, GC pauses, background processes

**Solution**:
- Increase `--benchmark-warmup` rounds
- Use `--benchmark-min-rounds` for more iterations
- Run on dedicated hardware for critical measurements

### False Regressions

**Cause**: Statistical noise in measurements

**Solution**:
- Increase `--benchmark-min-rounds` (default: 5)
- Use `--benchmark-sort=mean` for consistent sorting
- Check if regression is reproducible across runs

## Best Practices

### Writing Performance Tests

1. **Use pytest-benchmark**: Provides accurate timing and statistics
2. **Isolate operations**: Test single operations, not full workflows
3. **Use fixtures**: Set up test data efficiently
4. **Document targets**: Specify expected performance in docstrings
5. **Tag appropriately**: Use `@pytest.mark.benchmark` for clarity

### Example Test

```python
@pytest.mark.benchmark(group="my-feature")
def test_my_operation_performance(benchmark):
    """Test my_operation performance (< 1ms target)."""

    def operation():
        return my_operation(param1, param2)

    result = benchmark(operation)
    assert result is not None
```

### Maintaining Performance

1. **Run tests locally**: Before committing performance-sensitive changes
2. **Monitor CI results**: Review performance trends in PRs
3. **Update baselines**: After intentional performance improvements
4. **Investigate regressions**: Don't ignore performance warnings
5. **Document optimizations**: Explain why changes improve performance

## Related Documentation

- [Profiling Results](../../docs/performance/registration_profiling_results.md)
- [Architecture Design](../../docs/architecture/registration_indexed_architecture.md)
- [Batch Registration API](../../victor/tools/batch_registration.py)
- [Query Cache Implementation](../../victor/tools/query_cache.py)

## Contributing

When adding new performance tests:

1. Follow existing test structure and naming
2. Add to appropriate test group (registration, queries, etc.)
3. Document performance targets in docstring
4. Update this README with new test categories
5. Ensure tests are fast enough to run in CI (< 5 minutes total)

## License

Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>

Licensed under the Apache License, Version 2.0
