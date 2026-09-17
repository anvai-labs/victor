# Research Vertical

The Research vertical provides web research, fact-checking, literature synthesis, and report generation capabilities. It is designed to compete with Perplexity AI, Google Gemini Deep Research, and ChatGPT Browse.

## Overview

The Research vertical (`verticals/victor-research/victor_research/`) is specialized for web research tasks. Unlike coding assistants that focus on local codebases, the Research vertical searches the internet, fetches web pages, synthesizes information from multiple sources, and provides researched answers with citations.

### Key Use Cases

- **Deep Research**: Multi-source research with verification and synthesis
- **Fact-Checking**: Systematic claim verification with evidence evaluation
- **Literature Review**: Academic literature discovery and analysis
- **Competitive Analysis**: Market and competitor research
- **Quick Research**: Fast answers for simple queries
- **Report Generation**: Structured research documents with citations

## Available Tools

The Research vertical uses the following tools from `victor.tools.tool_names`:

| Tool | Description |
|------|-------------|
| `web_search` | Search the internet for information |
| `web_fetch` | Fetch and read content from URLs |
| `read` | Read local files and reports |
| `write` | Write research reports |
| `edit` | Modify existing documents |
| `ls` | List directory contents |
| `grep` | Search through documents |
| `code_search` | Search technical documentation |
| `overview` | Understand project structure |

## Available Workflows

### 1. Deep Research (`deep_research.yaml`)

Comprehensive multi-source research with verification:

```yaml
workflows:
  deep_research:
    nodes:
      # Planning
      - understand_query       # Analyze research question
      - create_search_plan     # Design search strategy

      # Source Discovery (Parallel)
      - parallel_search        # Fan-out to multiple search types
        - web_search           # General web search
        - academic_search      # Academic/scholarly sources
        - code_search          # Technical documentation

      # Validation
      - aggregate_sources      # Merge all results
      - validate_sources       # Check credibility
      - check_coverage         # Verify topic coverage
      - gap_analysis           # Identify missing information

      # Synthesis
      - synthesize             # Combine findings
      - generate_citations     # Format references (Compute)
      - review_synthesis       # HITL approval

      # Output
      - generate_report        # Create final document
```

**Key Features**:
- Parallel search across web, academic, and code sources
- Source credibility validation
- Coverage assessment with gap analysis
- Human-in-the-loop review gates
- Citation formatting (APA, MLA, Chicago, IEEE)

**Configuration**:
```yaml
coverage_threshold: 0.7        # 70% topic coverage before synthesis
citation_format: APA           # APA, MLA, Chicago, IEEE
max_search_queries: 10         # Per source type
source_quality_threshold: 0.6  # Minimum credibility score
hitl_timeout: 600s             # 10 min for gap decisions
report_max_tokens: 8000        # Final report length
```

### 2. Fact Check (`fact_check.yaml`)

Systematic fact verification with evidence evaluation:

```yaml
workflows:
  fact_check:
    nodes:
      # Claim Analysis
      - parse_claims           # Extract verifiable claims

      # Evidence Gathering (Parallel)
      - parallel_search
        - primary_sources      # Official/government sources
        - fact_check_sites     # Snopes, PolitiFact, etc.
        - news_archives        # Major news outlets

      # Evaluation
      - aggregate_evidence     # Combine all evidence
      - evaluate_evidence      # Assess quality and relevance
      - check_evidence_quality # Sufficiency check

      # Verdict
      - generate_verdicts      # Create fact-check verdicts
      - review_verdicts        # HITL review
      - generate_report        # Final fact-check report
```

**Verdict Categories**:
- **TRUE**: Claim is accurate
- **MOSTLY TRUE**: Substantially accurate with minor issues
- **MIXED**: Contains both true and false elements
- **MOSTLY FALSE**: Contains significant inaccuracies
- **FALSE**: Claim is inaccurate
- **UNVERIFIABLE**: Cannot be verified with available evidence

**Configuration**:
```yaml
source_credibility_threshold: 0.6
confidence_levels:
  high: ">0.85"
  medium: "0.6-0.85"
  low: "<0.6"
hitl_timeout: 600s             # 10 min for source decisions
review_timeout: 900s           # 15 min for verdict review
report_max_tokens: 6000
```

### 3. Quick Research (`deep_research.yaml`)

Fast research for simple queries:

```yaml
workflows:
  quick_research:
    nodes:
      - quick_search           # Fast web search
      - quick_summary          # 2-3 paragraph summary
```

**Configuration**:
```yaml
llm_config:
  temperature: 0.3
  model_hint: claude-3-sonnet  # Search
  model_hint: claude-3-haiku   # Summary (faster)
tool_budget: 20
```

## Stage Definitions

The Research vertical progresses through these stages:

| Stage | Description | Primary Tools |
|-------|-------------|---------------|
| `INITIAL` | Understanding the research question | `web_search`, `read`, `ls` |
| `SEARCHING` | Gathering sources and information | `web_search`, `web_fetch`, `grep` |
| `READING` | Deep reading and extraction | `web_fetch`, `read`, `code_search` |
| `SYNTHESIZING` | Combining and analyzing | `read`, `overview` |
| `WRITING` | Producing research output | `write`, `edit` |
| `VERIFICATION` | Fact-checking and validation | `web_search`, `web_fetch` |
| `COMPLETION` | Research complete with citations | (none) |

## Key Features

### Source Quality Assessment

Automatic evaluation of source credibility:

```yaml
credibility_factors:
  - author_authority       # Author credentials
  - publication_reputation # Source reputation
  - date_recency           # How recent
  - citation_count         # Academic citations
  - bias_indicators        # Potential bias markers
```

### Citation Management

Multiple citation formats supported:

```
# APA Format
Author, A. A. (Year). Title of work. Publisher.

# MLA Format
Author. "Title." Publisher, Year.

# Chicago Format
Author. Title. Place: Publisher, Year.

# IEEE Format
[1] A. Author, "Title," Publication, vol. X, pp. Y-Z, Year.
```

### Parallel Source Discovery

Simultaneous search across source types:

```python
parallel_nodes:
  - web_search      # General web: news, blogs, forums
  - academic_search # Google Scholar, PubMed, arXiv
  - code_search     # GitHub, Stack Overflow, docs
```

### Evidence Weighting

Evidence is weighted by multiple factors:

```yaml
evidence_weights:
  primary_source: 1.0      # Original/official sources
  peer_reviewed: 0.9       # Academic publications
  reputable_news: 0.7      # Major news outlets
  fact_check_sites: 0.8    # Established fact-checkers
  secondary_sources: 0.5   # Derivative reporting
```

### Capability Providers

The Research vertical provides these capabilities:

| Capability | Description |
|------------|-------------|
| `source_verification` | Source credibility validation |
| `citation_management` | Bibliography formatting |
| `research_quality` | Coverage assessment |
| `literature_analysis` | Paper relevance scoring |
| `fact_checking` | Evidence-based verdicts |

## Configuration Options

### Vertical Configuration

```python
from victor_research.assistant import ResearchAssistant

# Get system prompt
prompt = ResearchAssistant.get_system_prompt()

# Get tiered tools
tiered_tools = ResearchAssistant.get_tiered_tool_config()

# Access capability provider
capabilities = ResearchAssistant.get_capability_provider()

# Get capability configurations
configs = ResearchAssistant.get_capability_configs()
```

### Research Configuration

```yaml
# Source settings
sources:
  web:
    search_engines: [google, bing, duckduckgo]
    result_limit: 20
  academic:
    databases: [google_scholar, pubmed, arxiv]
    prefer_open_access: true
  fact_check:
    sites: [snopes, politifact, factcheck_org, reuters_fact_check]

# Quality settings
quality:
  min_sources: 3              # Minimum sources required
  cross_reference: true       # Verify across sources
  recency_preference: true    # Prefer recent sources
  recency_window_days: 365    # For time-sensitive topics

# Output settings
output:
  citation_format: APA
  include_methodology: true
  include_limitations: true
  max_report_length: 8000
```

### Workflow Parameters

```yaml
# Common workflow settings
llm_config:
  temperature: 0.3      # Factual accuracy
  max_tokens: 8000      # Long reports

tool_budget: 30         # Tool calls per node
timeout: 600            # Longer for research
```

## Example Usage

Use the public factory with the `research` vertical. Install the corresponding
`victor-research` package if it is not already available in your environment.

```python
import asyncio
from victor.framework import Agent

async def main():
    async with await Agent.create(
        vertical="research", provider="ollama", model="llama3.1:8b"
    ) as agent:
        result = await agent.run("Research the design tradeoffs of embedded vector databases")
        print(result.content)

asyncio.run(main())
```

For a named workflow supplied by the installed vertical, call
`await agent.run_workflow(workflow_name, context={...})` on the configured agent.
Use the installed package's workflow catalog to select a name and its expected input
keys. A bare workflow compiler does not create the agent runtime or provider.
See [Python API](../reference/api/python-api.md) and
[workflow execution](../tutorials/create-workflow.md).

## Integration with Other Verticals

The Research vertical integrates with:

- **RAG**: Build knowledge bases from research findings
- **Coding**: Research technical documentation and APIs
- **Data Analysis**: Statistical research and literature review

## File Structure

```
verticals/victor-research/victor_research/
├── assistant.py          # ResearchAssistant definition
├── capabilities.py       # Capability providers
├── mode_config.py        # Mode configurations
├── prompts.py            # Prompt templates
├── safety.py             # Safety checks for research
├── tool_dependencies.py  # Tool dependency configuration
├── workflows/
│   ├── deep_research.yaml  # Comprehensive research
│   └── fact_check.yaml     # Fact verification
├── handlers.py           # Compute handlers
├── escape_hatches.py     # Complex condition logic
├── rl.py                 # Reinforcement learning config
└── teams.py              # Multi-agent team specs
```

## Best Practices

1. **Use multiple sources**: Cross-reference claims across independent sources
2. **Verify credibility**: Check author authority and publication reputation
3. **Note recency**: Prefer recent sources for time-sensitive topics
4. **Cite everything**: Always attribute information to sources
5. **Acknowledge uncertainty**: Be transparent about limitations
6. **Distinguish facts from opinions**: Clearly separate factual claims
7. **Update findings**: Research can become outdated quickly

## Research Quality Standards

When conducting research:

- **Source Quality**: Prioritize authoritative sources (academic papers, official docs, reputable news)
- **Verification**: Cross-reference claims across multiple independent sources
- **Attribution**: Always cite sources with URLs or references
- **Objectivity**: Present balanced views, note controversies and limitations
- **Recency**: Prefer recent sources for time-sensitive topics

## Output Format

Research reports follow this structure:

1. **Executive Summary**: Key findings at a glance
2. **Introduction and Background**: Context for the research question
3. **Methodology**: Sources consulted and evaluation criteria
4. **Findings**: Organized by theme with citations
5. **Analysis and Discussion**: Interpretation of findings
6. **Conclusions**: Summary of insights
7. **Recommendations**: Action items if applicable
8. **References**: Full bibliography in requested format
9. **Limitations**: Areas needing further research

## Ethical Considerations

- **Never fabricate sources or statistics**
- **Acknowledge uncertainty when information is unclear**
- **Distinguish between facts, analysis, and opinions**
- **Update findings when new information emerges**
- **Respect copyright and fair use**
- **Disclose potential conflicts of interest**
