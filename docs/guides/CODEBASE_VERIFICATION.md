# Codebase Verification

**Semantic validation with false positive detection for codebase analysis.**

## Overview

The Codebase Verification module provides context-aware validation of codebase analysis results. It addresses common issues with automated code analysis tools:

- **False Positive Detection** - Identifies test code, compatibility shims, and intentional patterns
- **Documentation Cross-Reference** - Checks if issues are tracked in TECHNICAL_DEBT.adoc or roadmap
- **Temporal Analysis** - Distinguishes temporary issues (migration shims) from permanent problems
- **Severity Weighting** - Classifies issues by actual impact using configurable weights

## Installation

The verification module is included with Victor. No additional installation required.

```bash
# Verify the module is available
victor verify --help
```

## Quick Start

### CLI Usage

```bash
# Verify a single issue
victor verify issue cross_layer_dependency --file src/storage/lib.rs

# Verify with false positive analysis
victor verify issue global_mutable_state --file tests/test.rs --fp

# Batch verify from JSON file
victor verify batch issues.json --output verified.json

# Generate comprehensive report
victor verify report --output my_report.json

# List false positive patterns
victor verify fp-patterns

# Check documentation for technical debt
victor verify doc-check
```

### Python API Usage

```python
from pathlib import Path
from victor.tools.verification import (
    ClaimVerifier,
    FalsePositiveDetector,
    DocumentationCrossReference,
    TemporalContextAnalyzer,
    SeverityWeighting,
    ClaimIssue,
)

# Initialize components
project_root = Path(".")
verifier = ClaimVerifier(project_root=project_root)
fp_detector = FalsePositiveDetector()
crossref = DocumentationCrossReference(project_root=project_root)
temporal = TemporalContextAnalyzer(project_root=project_root)
weighting = SeverityWeighting()

# Create an issue
issue = ClaimIssue(
    issue_type="cross_layer_dependency",
    description="Storage depends on Index",
    file_path="src/storage/lib.rs",
    line_number=42,
)

# Verify the claim
result = await verifier.verify_claim(issue)
print(f"Verified: {result.is_grounded}")
print(f"Confidence: {result.confidence}")

# Check for false positives
is_fp, reason, confidence = fp_detector.is_likely_false_positive(issue)
if is_fp:
    print(f"False positive: {reason} ({confidence:.1%})")

# Check documentation
if crossref.is_tracked_debt(issue):
    print("Issue is tracked in TECHNICAL_DEBT.adoc")

# Analyze temporal context
temporal_ctx = temporal.analyze_issue_temporal_context(issue)
print(f"Temporal nature: {temporal_ctx['temporal_nature']}")

# Calculate severity
score, severity = weighting.score_and_classify(issue)
print(f"Severity: {severity.value} (score: {score:.2f})")
```

### Batch Verification

```python
from victor.tools.verification.report_generator import (
    VerificationReportGenerator,
    ReportFormat,
)

# Create report generator
generator = VerificationReportGenerator(
    project_root=Path("."),
    enable_fp_detection=True,
    enable_doc_crossref=True,
    enable_temporal_analysis=True,
    enable_severity_weighting=True,
)

# Generate report from issues
issues = [
    {"issue_type": "cross_layer_dependency", "file_path": "src/storage/lib.rs"},
    {"issue_type": "global_mutable_state", "file_path": "tests/test.rs"},
]

report = await generator.generate_report(issues)

# Save in different formats
generator.save_report(report, "report.json", ReportFormat.JSON)
generator.save_report(report, "report.md", ReportFormat.MARKDOWN)
generator.save_report(report, "report.txt", ReportFormat.CONSOLE)

# Access summary
print(f"Total issues: {report.summary.total_issues}")
print(f"Genuine issues: {report.summary.genuine_issues}")
print(f"False positives: {report.summary.false_positives}")
```

## Verification Components

### ClaimVerifier

Core verification with evidence collection and confidence scoring.

```python
from victor.tools.verification import ClaimVerifier, ClaimIssue

verifier = ClaimVerifier(project_root=Path("."))
result = await verifier.verify_claim(issue)

# Result fields
result.is_grounded     # bool: Whether claim is verified
result.confidence      # float: 0.0-1.0 confidence score
result.evidence        # dict: Collected evidence
result.reason          # str: Human-readable explanation
```

### FalsePositiveDetector

Pattern-based detection of common false positive categories.

**Pattern Categories:**

| Category | Description | Example |
|----------|-------------|---------|
| `test_global_state` | Test-specific global state | `#[cfg(test)] static ...` |
| `test_attribute` | Test functions | `#[test] fn test_...` |
| `test_file_path` | Files in test directories | `tests/test_*.rs` |
| `compatibility_shim` | Intentional re-exports | `Root compatibility re-exports` |
| `documented_debt` | Tracked technical debt | `TD-CROSS-LAYER: ...` |
| `generated_code` | Auto-generated code | `DO NOT EDIT`, `Generated by` |
| `fixture_data` | Test fixtures | `fixtures/`, `mock_data/` |
| `intentional_pattern` | Known intentional patterns | Custom patterns |

```python
from victor.tools.verification import FalsePositiveDetector

detector = FalsePositiveDetector()
is_fp, reason, confidence = detector.is_likely_false_positive(issue)

# Filter a batch of issues
filtered_issues = detector.filter_issues(issues_list, confidence_threshold=0.7)
```

### DocumentationCrossReference

Cross-references issues with project documentation.

```python
from victor.tools.verification import DocumentationCrossReference

crossref = DocumentationCrossReference(project_root=Path("."))

# Check if tracked in technical debt
if crossref.is_tracked_debt(issue):
    markers = crossref.get_tech_debt_markers()
    print(f"Found {len(markers)} debt markers")

# Check roadmap alignment
if crossref.check_roadmap_alignment(issue):
    print("Issue is addressed in roadmap")

# Get all documentation references
doc_refs = crossref.get_doc_references(issue.model_dump())
```

**Documentation Files:**
- `docs/10-quality/TECHNICAL_DEBT.adoc` - Technical debt tracking
- `docs/_internal/roadmap.md` - Project roadmap
- `docs/10-quality/known-issues.md` - Known issues catalog

### TemporalContextAnalyzer

Analyzes temporal context to classify issues as temporary or permanent.

```python
from victor.tools.verification import TemporalContextAnalyzer, TemporalNature

temporal = TemporalContextAnalyzer(project_root=Path("."))
context = temporal.analyze_issue_temporal_context(issue)

# Context fields
context['temporal_nature']   # TEMPORARY, PERMANENT, or UNKNOWN
context['file_age_days']     # Age of file in days
context['has_removal_plan']  # Whether removal plan exists
context['recent_changes']    # Recent git changes
```

**Temporal Classifications:**

| Classification | Description | Example |
|----------------|-------------|---------|
| `TEMPORARY` | Likely to resolve soon | Migration shims, WIP code |
| `PERMANENT` | Long-standing issue | Architectural problems |
| `UNKNOWN` | Cannot determine | Insufficient context |

### SeverityWeighting

Weights issues by actual impact using configurable factors.

```python
from victor.tools.verification import SeverityWeighting, SeverityLevel

weighting = SeverityWeighting()

# Calculate score and classify
score, severity = weighting.score_and_classify(issue)

# Severity levels
# CRITICAL - Security issues, data loss risk
# HIGH - Performance, compilation time
# MEDIUM - Maintainability, code quality
# LOW - Style, minor issues
# INFO - Observations, suggestions
```

**Impact Factors:**

| Factor | Weight | Description |
|--------|--------|-------------|
| `compilation_time` | 0.3 | Affects build time |
| `runtime_performance` | 0.4 | Runtime overhead |
| `maintainability` | 0.2 | Code maintenance |
| `security` | 0.5 | Security vulnerabilities |
| `test_reliability` | 0.15 | Test stability |

## CLI Commands

### victor verify issue

Verify a single codebase issue.

```bash
victor verify issue ISSUE_TYPE [options]

Options:
  --file, -f PATH      Path to file
  --line, -l NUMBER    Line number
  --desc, -d TEXT      Issue description
  --evidence, -e       Show collected evidence
  --fp                 Show false positive analysis
  --root, -r PATH      Project root directory
```

**Examples:**

```bash
# Basic verification
victor verify issue cross_layer_dependency --file src/storage/lib.rs

# With evidence and FP analysis
victor verify issue security_vulnerability \
  --file src/auth.rs \
  --desc "Buffer overflow" \
  --evidence --fp

# Test code pattern
victor verify issue global_mutable_state \
  --file tests/test.rs \
  --fp
```

### victor verify batch

Verify multiple issues from a JSON file.

```bash
victor verify batch INPUT_FILE [options]

Options:
  --output, -o PATH    Output JSON file
  --min-confidence, -c NUMBER  Minimum confidence (default: 0.5)
  --root, -r PATH      Project root directory
```

**Input Format:**

```json
[
  {
    "issue_type": "cross_layer_dependency",
    "description": "Storage depends on Index",
    "file_path": "src/storage/lib.rs"
  },
  {
    "issue_type": "global_mutable_state",
    "file_path": "tests/test.rs",
    "line_number": 42
  }
]
```

### victor verify report

Generate comprehensive verification report.

```bash
victor verify report [options]

Options:
  --output, -o PATH    Output file (default: verification_report.json)
  --root, -r PATH      Project root directory
```

**Report Contents:**

- Component status (FP detection, doc crossref, temporal analysis)
- False positive pattern counts
- Technical debt marker counts
- Git availability for temporal analysis

### victor verify fp-patterns

List all false positive detection patterns.

```bash
victor verify fp-patterns [options]

Options:
  --custom, -c         Show custom patterns only
```

### victor verify doc-check

Check project documentation for technical debt tracking.

```bash
victor verify doc-check [options]

Options:
  --root, -r PATH      Project root directory
```

**Displays:**
- Technical debt markers found
- Roadmap priorities by category

## Tool Integration

The verification module integrates with Victor's tool system:

```python
from victor.framework import Agent

agent = await Agent.create()

# Use the codebase_verify tool
result = await agent.run(
    "Analyze this codebase for cross-layer dependencies",
    tools=["codebase_verify"],
)

# Batch verification
result = await agent.run(
    "Verify these issues and filter false positives",
    tools=["codebase_verify_batch"],
)
```

**Tool Parameters:**

```python
# codebase_verify
{
    "query": str,                    # Issue description
    "enable_fp_detection": bool,     # Enable false positive detection
    "enable_doc_crossref": bool,     # Enable documentation cross-reference
    "enable_temporal_analysis": bool,# Enable temporal analysis
    "enable_severity_weighting": bool # Enable severity weighting
}

# codebase_verify_batch
{
    "issues": list[dict],            # List of issues to verify
    "min_confidence": float,         # Minimum confidence threshold
    "enable_fp_detection": bool,     # Enable false positive detection
    "enable_doc_crossref": bool,     # Enable documentation cross-reference
    "enable_temporal_analysis": bool,# Enable temporal analysis
    "enable_severity_weighting": bool # Enable severity weighting
}
```

## Configuration

### Custom False Positive Patterns

Add custom patterns to the detector:

```python
from victor.tools.verification import FalsePositiveDetector

detector = FalsePositiveDetector()

# Add custom pattern
detector.FALSE_POSITIVE_PATTERNS["custom_intentional"] = [
    re.compile(r"@intentional-pattern"),
    re.compile(r"# INTENTIONAL: .+"),
]
```

### Custom Severity Weights

Override default impact weights:

```python
from victor.tools.verification import SeverityWeighting

weighting = SeverityWeighting()

# Customize weights
weighting.IMPACT_WEIGHTS["compilation_time"] = 0.5  # Increase importance
weighting.IMPACT_WEIGHTS["style"] = 0.05            # Decrease importance
```

### Documentation Paths

Customize documentation file locations:

```python
from victor.tools.verification import DocumentationCrossReference
from pathlib import Path

crossref = DocumentationCrossReference(project_root=Path("."))

# Override default paths
crossref.tech_debt_doc = Path("docs/quality/debt.adoc")
crossref.roadmap_doc = Path("docs/planning/roadmap.md")
crossref.known_issues_doc = Path("docs/issues/known.md")
```

## Best Practices

1. **Enable All Features** - Use full verification for comprehensive analysis
2. **Filter High Confidence** - Set appropriate confidence thresholds for batch operations
3. **Document Technical Debt** - Track issues in TECHNICAL_DEBT.adoc for cross-reference
4. **Review False Positives** - Always review FP analysis before dismissing issues
5. **Consider Temporal Context** - Temporary issues may not need immediate action
6. **Weight by Impact** - Prioritize CRITICAL and HIGH severity issues

## Troubleshooting

### False Positives Not Detected

- Ensure pattern regexes match your codebase conventions
- Add custom patterns for project-specific intentional code
- Check file path patterns match your directory structure

### Documentation Cross-Reference Fails

- Verify TECHNICAL_DEBT.adoc exists at `docs/10-quality/TECHNICAL_DEBT.adoc`
- Check markdown parsing for roadmap.md
- Ensure TD-* markers follow the correct format

### Temporal Analysis Returns UNKNOWN

- Ensure git is available and repository is initialized
- Check file path is relative to project root
- Verify git history is accessible

## API Reference

### ClaimIssue

```python
class ClaimIssue(BaseModel):
    issue_type: str                    # Type of issue
    description: Optional[str]         # Human-readable description
    file_path: Optional[str]           # Path to file
    line_number: Optional[int]         # Line number
    snippet: Optional[str]             # Code snippet
    severity: Optional[SeverityLevel]  # Initial severity
    category: Optional[IssueCategory]  # Issue category
    metadata: Dict[str, Any]           # Additional metadata
```

### EnhancedClaimResult

```python
class EnhancedClaimResult(BaseModel):
    is_grounded: bool                  # Whether claim is verified
    confidence: float                  # 0.0-1.0 confidence
    evidence: Dict[str, Any]           # Supporting evidence
    reason: str                        # Explanation
    false_positive_risk: float         # 0.0-1.0 FP risk
    severity: Optional[SeverityLevel]  # Classified severity
    temporal_nature: Optional[TemporalNature]  # Temporal classification
    doc_references: List[str]          # Documentation references
    category: Optional[IssueCategory]  # Issue category
```

### SeverityLevel

```python
class SeverityLevel(str, Enum):
    CRITICAL = "critical"  # Security, data loss
    HIGH = "high"          # Performance, compilation
    MEDIUM = "medium"      # Maintainability
    LOW = "low"            # Style, minor issues
    INFO = "info"          # Observations
```

### TemporalNature

```python
class TemporalNature(str, Enum):
    TEMPORARY = "temporary"  # Likely to resolve soon
    PERMANENT = "permanent"  # Long-standing issue
    UNKNOWN = "unknown"      # Cannot determine
```

### IssueCategory

```python
class IssueCategory(str, Enum):
    ARCHITECTURAL = "architectural"      # Cross-layer dependencies
    CODE_QUALITY = "code_quality"        # Style, complexity
    PERFORMANCE = "performance"          # Runtime, compilation
    SECURITY = "security"                # Vulnerabilities
    MAINTAINABILITY = "maintainability"  # Documentation, modularity
    COMPATIBILITY = "compatibility"      # Shims, re-exports
    TESTING = "testing"                  # Test-specific patterns
    DOCUMENTATION = "documentation"      # Missing docs
```

## See Also

- [Tool Reference](tool-reference.md) - General tool usage
- [Development Guide](../development/index.md) - Extension development
- [Testing](../development/testing.md) - Testing patterns
