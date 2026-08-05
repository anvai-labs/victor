# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark runners for industry-standard evaluation datasets.

All runners load REAL benchmark data from HuggingFace and execute
REAL tests. Results are not simulated - they represent actual test
execution against actual benchmark problems.

Supported benchmarks:
- SWE-bench: Real-world GitHub issues from Python repositories
- HumanEval: Code generation from docstrings (OpenAI)
- MBPP: Mostly Basic Python Problems (Google Research)

Framework comparison:
- Compare Victor against Aider, Claude Code, Cursor, and others
- Uses published benchmark results for standardized comparison
"""

from victor.evaluation.benchmarks.swe_bench import (
    HumanEvalRunner,
    MBPPRunner,
    SWEBenchRunner,
)
from victor.evaluation.benchmarks.external_agentic import (
    ExternalAgenticBenchmarkRunner,
)
from victor.evaluation.benchmarks.browser_tasks import (
    BrowserTaskBenchmarkRunner,
)
from victor.evaluation.benchmarks.deep_research import (
    DeepResearchBenchmarkRunner,
)
from victor.evaluation.benchmarks.framework_comparison import (
    DEFAULT_FIXTURE_SET_ROOT,
    ComparisonMetrics,
    ComparisonReport,
    FixtureBenchmarkDescriptor,
    FixtureSetDescriptor,
    FixtureSetVerificationResult,
    build_fixture_benchmark_catalog,
    build_publication_corpus_readiness,
    build_publication_stable_run_summary,
    canonicalize_fixture_benchmark_name,
    Framework,
    FrameworkCapabilities,
    FrameworkResult,
    FRAMEWORK_CAPABILITIES,
    fixture_benchmark_matches,
    PUBLISHED_RESULTS,
    compute_metrics_from_result,
    compute_metrics_from_saved_result,
    create_comparison_report,
    create_comparison_report_from_fixture_manifest,
    create_comparison_report_from_saved_result,
    create_comparison_report_from_saved_results,
    discover_fixture_benchmarks,
    discover_fixture_sets,
    get_published_result,
    load_framework_result_from_file,
    resolve_fixture_sets_for_benchmark,
    resolve_fixture_set_names,
    resolve_fixture_benchmark_publication_manifests,
    save_fixture_benchmark_catalog,
    save_fixture_benchmark_publication_bundle,
    save_stable_run_publication_bundle,
    save_comparison_report_bundle,
    verify_fixture_sets,
)

__all__ = [
    # Runners
    "HumanEvalRunner",
    "MBPPRunner",
    "SWEBenchRunner",
    "ExternalAgenticBenchmarkRunner",
    "BrowserTaskBenchmarkRunner",
    "DeepResearchBenchmarkRunner",
    # Framework comparison
    "DEFAULT_FIXTURE_SET_ROOT",
    "ComparisonMetrics",
    "ComparisonReport",
    "FixtureBenchmarkDescriptor",
    "FixtureSetDescriptor",
    "FixtureSetVerificationResult",
    "build_fixture_benchmark_catalog",
    "build_publication_corpus_readiness",
    "build_publication_stable_run_summary",
    "canonicalize_fixture_benchmark_name",
    "Framework",
    "FrameworkCapabilities",
    "FrameworkResult",
    "FRAMEWORK_CAPABILITIES",
    "fixture_benchmark_matches",
    "PUBLISHED_RESULTS",
    "compute_metrics_from_result",
    "compute_metrics_from_saved_result",
    "create_comparison_report",
    "create_comparison_report_from_fixture_manifest",
    "create_comparison_report_from_saved_result",
    "create_comparison_report_from_saved_results",
    "discover_fixture_benchmarks",
    "discover_fixture_sets",
    "get_published_result",
    "load_framework_result_from_file",
    "resolve_fixture_sets_for_benchmark",
    "resolve_fixture_set_names",
    "resolve_fixture_benchmark_publication_manifests",
    "save_fixture_benchmark_catalog",
    "save_fixture_benchmark_publication_bundle",
    "save_stable_run_publication_bundle",
    "save_comparison_report_bundle",
    "verify_fixture_sets",
]
