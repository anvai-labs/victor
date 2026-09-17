# RAG Vertical

The RAG (Retrieval-Augmented Generation) vertical provides document ingestion, vector search, and question-answering capabilities. It enables building knowledge bases from documents and answering questions with source citations.

## Overview

The RAG vertical (`verticals/victor-rag/victor_rag/`) implements a complete RAG pipeline with document processing, semantic chunking, embedding generation, and hybrid search. It uses LanceDB for embedded vector storage (no server required) and supports multiple embedding providers.

### Key Use Cases

- **Document Q&A**: Answer questions based on ingested documents with citations
- **Knowledge Base Management**: Build and maintain searchable document collections
- **Semantic Search**: Find relevant information using natural language queries
- **Multi-Turn Conversations**: Contextual Q&A with conversation history
- **Agentic RAG**: Complex queries requiring multi-hop reasoning

## Available Tools

The RAG vertical provides specialized tools for document operations:

| Tool | Description |
|------|-------------|
| `rag_ingest` | Ingest documents into the vector store |
| `rag_search` | Search for relevant chunks in the knowledge base |
| `rag_query` | Query with automatic context retrieval and answer generation |
| `rag_list` | List indexed documents |
| `rag_delete` | Delete documents from the index |
| `rag_stats` | Get store statistics (count, size, etc.) |
| `read` | Read file contents for ingestion |
| `ls` | List directory contents |
| `web_fetch` | Fetch web content for ingestion |
| `shell` | Execute document processing commands |

## Available Workflows

### 1. Document Ingestion (`ingest.yaml`)

Comprehensive pipeline for processing and indexing documents:

```yaml
workflows:
  document_ingest:
    nodes:
      - discover_documents     # Find files matching patterns
      - filter_documents       # Deduplicate and filter
      - parallel_parse         # Parse PDF, DOCX, MD, code (parallel)
      - validate_content       # Check encoding and quality
      - chunking               # Split into semantic chunks
      - extract_metadata       # NER, topics, dates (parallel)
      - generate_embeddings    # Batch embedding generation
      - ingest_vectors         # Store in vector database
      - generate_report        # Create ingestion summary
```

**Key Features**:
- Multi-format parsing: PDF, DOCX, Markdown, Text, Code
- Intelligent chunking with semantic boundaries
- Metadata extraction (entities, topics, dates)
- Batch embedding with rate limit handling
- Progress tracking and error recovery

**Configuration**:
```yaml
chunk_size: 512          # Tokens per chunk
chunk_overlap: 50        # Overlap between chunks
embedding_model: "text-embedding-3-small"
batch_size: 100          # Chunks per API call
```

### 2. RAG Query (`query.yaml`)

Answer questions using retrieved context with citations:

```yaml
workflows:
  rag_query:
    nodes:
      - analyze_query        # Understand query intent
      - expand_query         # Generate search variants
      - parallel_search      # Dense + sparse + entity search
      - merge_results        # RRF fusion
      - rerank               # Semantic relevance scoring
      - check_coverage       # Evaluate context sufficiency
      - generate_answer      # Synthesize with citations
      - verify_answer        # Fact-check against sources
```

**Key Features**:
- Hybrid search (dense vectors + BM25 + entity matching)
- Reciprocal Rank Fusion for result merging
- LLM-based reranking for relevance
- Answer verification and citation checking
- Confidence scoring

### 3. Multi-Turn Conversation (`query.yaml`)

Contextual Q&A maintaining conversation history:

```yaml
workflows:
  conversation:
    nodes:
      - load_history       # Load previous turns
      - contextualize      # Resolve references
      - parallel_search    # Search with context
      - generate           # Contextual answer
      - save_turn          # Persist conversation
```

### 4. Agentic RAG (`query.yaml`)

Complex queries requiring multi-step reasoning:

```yaml
workflows:
  agentic_rag:
    nodes:
      - plan_approach      # Plan sub-questions
      - execute_plan       # Multi-step search
      - synthesize         # Combine findings
      - generate_response  # Final answer
```

### 5. Index Maintenance (`query.yaml`)

Keep the index healthy and optimized:

```yaml
workflows:
  maintenance:
    nodes:
      - analyze_index      # Collect statistics
      - cleanup            # Remove orphaned/stale entries
      - optimize           # Compact and rebuild index
      - generate_report    # Maintenance summary
```

## Stage Definitions

The RAG vertical progresses through these stages:

| Stage | Description | Primary Tools |
|-------|-------------|---------------|
| `INITIAL` | Ready to accept queries | `rag_search`, `rag_query`, `rag_list`, `rag_stats` |
| `INGESTING` | Processing documents | `rag_ingest`, `read`, `ls`, `web_fetch` |
| `SEARCHING` | Searching knowledge base | `rag_search`, `rag_query` |
| `QUERYING` | Processing with context | `rag_query` |
| `SYNTHESIZING` | Generating answers | (LLM only) |

## Key Features

### LanceDB Vector Storage

Victor's RAG uses LanceDB, an embedded vector database that requires no server:

```python
# Storage location
~/.victor/vectors/  # Default location
.victor/vectors/    # Project-specific

# Features
- Embedded (no server process)
- HNSW indexing for fast similarity search
- Hybrid search support
- Automatic persistence
```

### Supported Document Formats

| Format | Parser | Notes |
|--------|--------|-------|
| PDF | pdfplumber | Tables, images, optional OCR |
| DOCX | python-docx | Preserves heading hierarchy |
| Markdown | mistune | Extracts header structure |
| Text | Direct read | Plain text files |
| Code | Tree-sitter | AST-aware chunking |
| RST | docutils | reStructuredText |

### Chunking Strategies

Different strategies for different content types:

```yaml
strategies:
  technical_docs: section_based    # Split on headers
  code_files: ast_boundaries       # Function/class boundaries
  narratives: semantic_paragraphs  # Paragraph breaks
  tables: row_preserving           # Keep table rows together
```

### Embedding Providers

Supports multiple embedding models:

| Provider | Model | Dimensions |
|----------|-------|------------|
| OpenAI | text-embedding-3-small | 1536 |
| OpenAI | text-embedding-3-large | 3072 |
| Local | all-MiniLM-L6-v2 | 384 |
| Local | all-mpnet-base-v2 | 768 |
| Cohere | embed-english-v3.0 | 1024 |

### Hybrid Search

Combines multiple retrieval methods:

1. **Dense Search**: Vector similarity (cosine/dot product)
2. **Sparse Search**: BM25 keyword matching
3. **Entity Search**: Structured metadata filtering
4. **Result Fusion**: Reciprocal Rank Fusion (RRF)

```python
# RRF Formula
score = sum(1 / (k + rank)) for each result set
# k = 60 (default smoothing constant)
```

## Configuration Options

### Vertical Configuration

```python
from victor_rag.assistant import RAGAssistant

# Get system prompt
prompt = RAGAssistant.get_system_prompt()

# Get tiered tools
tiered_tools = RAGAssistant.get_tiered_tool_config()

# Access capability provider
capabilities = RAGAssistant.get_capability_provider()
```

### Ingestion Configuration

```yaml
# Default ingestion settings
chunk_size: 512               # Tokens per chunk
chunk_overlap: 50             # 10% overlap
embedding_model: "text-embedding-3-small"
embedding_dimensions: 1536
batch_size: 100               # Chunks per API call
ocr_enabled: false            # Enable for scanned PDFs
retention_days: 365           # Document retention
```

### Query Configuration

```yaml
# Default query settings
top_k: 10                     # Chunks to retrieve
rerank_top_k: 5               # After reranking
similarity_threshold: 0.7     # Minimum relevance
max_context_tokens: 4000      # Context window
answer_max_tokens: 2000       # Answer length
temperature: 0.3              # Factual answers
citation_style: inline        # [1], [2] style
```

### Service Configuration

```yaml
# SQLite project database
project_db:
  path: $project/.victor/project.db
  journal_mode: WAL
  cache_size: 10000

# Vector store
vector_store:
  type: lancedb
  path: $project/.victor/vectors
```

## Example Usage

Use the public factory with the `rag` vertical. Install the corresponding
`victor-rag` package if it is not already available in your environment.

```python
import asyncio
from victor.framework import Agent

async def main():
    async with await Agent.create(
        vertical="rag", provider="ollama", model="llama3.1:8b"
    ) as agent:
        result = await agent.run("Find documentation relevant to authentication in this project")
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

The RAG vertical integrates with:

- **Coding**: Search code documentation and comments
- **Research**: Build knowledge bases from research materials
- **DevOps**: Query infrastructure documentation

## File Structure

```
verticals/victor-rag/victor_rag/
├── assistant.py          # RAGAssistant vertical definition
├── capabilities.py       # Capability providers
├── mode_config.py        # Mode configurations
├── prompts.py            # Prompt templates
├── safety.py             # Safety checks for RAG
├── tool_dependencies.py  # Tool dependency configuration
├── tools/
│   ├── ingest.py         # rag_ingest tool
│   ├── search.py         # rag_search tool
│   ├── query.py          # rag_query tool
│   └── ...
├── workflows/
│   ├── ingest.yaml       # Document ingestion workflow
│   └── query.yaml        # Query and conversation workflows
├── handlers.py           # Compute handlers
├── escape_hatches.py     # Complex condition logic
├── rl.py                 # Reinforcement learning config
└── teams.py              # Multi-agent team specs
```

## Best Practices

1. **Chunk size matters**: Smaller chunks (256-512 tokens) for specific facts, larger (1000+) for summaries
2. **Use hybrid search**: Combine dense and sparse for best results
3. **Enable reranking**: LLM reranking significantly improves relevance
4. **Verify answers**: Always verify generated answers against sources
5. **Maintain the index**: Regular maintenance prevents degradation
6. **Handle edge cases**: Plan for no-context and low-confidence scenarios
7. **Cite sources**: Always attribute information to sources

## Performance Considerations

- **Embedding costs**: Batch API calls to reduce costs
- **Index size**: Monitor disk usage for large collections
- **Query latency**: Use appropriate top_k values
- **Memory usage**: LanceDB is memory-efficient for most use cases
- **Concurrent queries**: LanceDB handles concurrent reads well
