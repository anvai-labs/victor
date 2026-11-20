# Tool Reference

**Quick reference for Victor's 34 tool modules.**

## Tool Categories

| Category | Tools | Cost | Description |
|----------|-------|------|-------------|
| **Filesystem** | read, write, edit, grep, ls, overview | FREE | File operations |
| **Git** | status, log, diff, commit, branch | LOW | Version control |
| **Search** | code_search, symbol, refs | LOW | Code navigation |
| **Execution** | shell, python, docker_exec | MEDIUM | Run code |
| **Web** | web_search, web_fetch | MEDIUM | Network requests |
| **Analysis** | review, test_generation | HIGH | LLM-intensive |
| **Verification** | codebase_verify, codebase_verify_batch | LOW | Semantic validation |

## Core Tools

### Filesystem

| Tool | Description | Example |
|------|-------------|---------|
| `read` | Read file contents | `read src/main.py` |
| `write` | Write/create file | `write output.txt "hello"` |
| `edit` | Edit file with regex | `edit main.py s/foo/bar/` |
| `grep` | Search file contents | `grep "TODO" **/*.py` |
| `ls` | List directory | `ls src/` |
| `overview` | Codebase summary | `overview` |

### Git

| Tool | Description | Example |
|------|-------------|---------|
| `git_status` | Show repo status | `git status` |
| `git_log` | Show commit history | `git log -n 10` |
| `git_diff` | Show changes | `git diff HEAD~1` |
| `git_commit` | Commit changes | `git commit -m "fix"` |
| `git_branch` | Manage branches | `git branch feature` |

### Search

| Tool | Description | Example |
|------|-------------|---------|
| `code_search` | Semantic search | `code_search "authentication"` |
| `symbol` | Find symbol definition | `symbol MyClass` |
| `refs` | Find references | `refs my_function` |

### Execution

| Tool | Description | Example |
|------|-------------|---------|
| `shell` | Run shell command | `shell pytest tests/` |
| `python` | Run Python code | `python print(1+1)` |
| `docker_exec` | Run in container | `docker_exec pytest` |

### Web

| Tool | Description | Example |
|------|-------------|---------|
| `web_search` | Search web (DuckDuckGo by default; Tavily/Brave/Serper with a key) | `web_search "Python async"` |
| `web_fetch` | Readable page fetch (title + main text, size cap, content-type guard) | `web_fetch https://example.com` |

**Search providers (FOU-A5, optional, key-gated):** `web_search(provider=...)` accepts
`auto` (default), `duckduckgo` (no key), `tavily`, `brave`, or `serper`. A provider is
only offered when its key resolves; `auto` falls back to DuckDuckGo when no key is set.
One-line setup: `export TAVILY_API_KEY=...` (or `victor keys --set-service tavily`;
likewise `BRAVE_SEARCH_API_KEY` / `SERPER_API_KEY`).

## Tool Tiers

| Tier | Tools | Impact | When to Use |
|------|-------|--------|-------------|
| **FREE** | Filesystem, grep, ls | None | Always available |
| **LOW** | Git, search | Minimal | Most tasks |
| **MEDIUM** | Shell, web, docker | Moderate | When needed |
| **HIGH** | LLM analysis | Significant | Complex tasks |

## Tool Selection

### Strategy Comparison

| Strategy | Description | Speed | Accuracy |
|----------|-------------|-------|----------|
| `keyword` | Exact name matching | ⚡⚡⚡ | Low |
| `semantic` | Embedding similarity | ⚡ | High |
| `hybrid` | 70% semantic + 30% keyword | ⚡⚡ | High |

### Configuration

```yaml
# ~/.victor/config.yaml
tool_selection:
  strategy: hybrid  # keyword | semantic | hybrid
  semantic_threshold: 0.7
```

## Progressive Tools

Some tools support parameter escalation:

| Tool | Initial | Progressive | Max |
|------|---------|-------------|-----|
| `code_search` | 5 results | +10 per call | 100 |
| `grep` | 50 matches | +50 | 500 |

## Aliases

| Alias | Resolves To |
|-------|-------------|
| `shell` | `bash` → `zsh` → `sh` |
| `grep` | `rg` → `grep` |

## See Also

- [Tool Catalog](../reference/tools/catalog.md) - Complete tool documentation
- [Tool Calling Formats](development/TOOL_CALLING_FORMATS.md) - Provider formats
