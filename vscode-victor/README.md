# Victor AI - VS Code Extension

AI-powered agentic AI framework with multi-provider support, semantic code search, and 33 tool modules for professional development workflows.

[![GitHub](https://img.shields.io/badge/GitHub-anvai--labs%2Fvictor-blue)](https://github.com/anvai-labs/victor)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/anvai-labs/victor/blob/main/LICENSE)

## Features

### Background Agents (NEW in v0.2.0)
- Run AI agents asynchronously with real-time progress tracking
- Agent sidebar shows active tasks with tool call details
- WebSocket-based real-time updates
- Cancel running agents, view output
- Status bar indicator shows active agent count

### Terminal Agent (NEW in v0.2.0)
- AI-assisted terminal command execution
- Suggest commands based on natural language intent
- Dangerous command detection with approval workflow
- Command history and approval tracking

### Smart Paste (NEW in v0.2.0)
- Automatically adapt pasted code to current context
- Adjusts indentation, naming conventions
- Adds missing imports based on file context
- Preview adaptations before applying

### Multi-file Composer (NEW in v0.2.0)
- Complex multi-file edits in one prompt
- Add multiple files to composer context
- Describe changes in natural language
- Preview all changes before applying
- Selective application of changes

### Autonomy Controls (NEW in v0.2.0)
- Control how much freedom Victor has
- Autonomy levels: Manual, Semi-auto, Auto
- Configurable approval thresholds for terminal/file operations
- Dangerous command pattern detection

### @workspace Context (NEW in v0.2.0)
- Modern repo-aware context
- `@workspace`: Project structure, configs, open files
- `@selection`: Current editor selection
- `@terminal`: Active terminal context
- Enhanced `@git`, `@problems`, `@file:`, `@folder:`

### Inline Edit with Diff Preview
- Edit selected code with AI suggestions
- Ghost text preview of changes
- Show diff before applying
- Accept/reject with Tab/Escape

### FIM Completions
- Fill-in-the-Middle support for better inline suggestions
- Includes code after cursor for context-aware completions

### Automatic Server Management
- **Auto-start**: Server starts automatically when VS Code opens
- **Health monitoring**: Extension monitors server health
- **Status indicator**: See server status in status bar
- **Easy control**: Start/stop/restart via command palette

### Chat Panel
- Interactive chat with AI assistant
- Code explanation, refactoring, and generation
- Tool call visualization
- Streaming responses

### Inline Completions
- AI-powered code completions
- Context-aware suggestions
- Multi-language support

### Semantic Code Search
- Natural language code search
- Find code by describing what it does
- Jump to results instantly

### Context Management
- `@file:path` - Include file contents
- `@folder:path` - Include folder structure
- `@problems` - Include workspace diagnostics
- `@git` - Include git status

### Agent Modes
- **Build** - Full implementation mode with all tools
- **Plan** - Read-only analysis mode
- **Explore** - Code navigation and understanding mode

### Multi-Provider Support
- Anthropic (Claude Opus 4.5, Sonnet 4)
- OpenAI (GPT-4 Turbo, GPT-4o)
- Google (Gemini 2.0 Flash)
- Ollama (Qwen 2.5, Llama 3.1)
- LMStudio (Local models)
- vLLM (Self-hosted)
- xAI (Grok)

## Installation

### Prerequisites

1. **Python 3.10+** installed
2. **Victor** installed:
   ```bash
   pip install -e ".[dev]"
   victor init
   ```

### Install Extension

1. Install this extension from VS Code Marketplace
2. The server will start automatically on first activation

Or manually from source:
```bash
cd vscode-victor
npm install
npm run compile
```

### Connecting to a secured server

If your Victor server sets `VICTOR_SERVER_API_KEY`:
- In VS Code settings, set `victor.serverApiKey` to the same token.
- Set `victor.serverPort` or `victor.serverUrl` to point at your server (default `http://localhost:8765` if you started `victor serve --port 8765`).
- The extension will send `Authorization: Bearer <token>`, prefetch a signed `session_token`, and reuse it for WebSocket reconnects automatically.

## Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| Victor: Open Chat | `Ctrl+Shift+V` | Open chat panel |
| Victor: Explain | `Ctrl+Shift+E` | Explain selected code |
| Victor: Search | `Ctrl+Shift+S` | Semantic code search |
| Victor: Start Agent | `Cmd/Ctrl+Shift+A` | Start background agent |
| Victor: Suggest Command | `Cmd/Ctrl+Shift+T` | AI terminal command suggestion |
| Victor: Smart Paste | `Cmd/Ctrl+Shift+P` | Context-aware paste |
| Victor: Open Composer | `Cmd/Ctrl+Shift+C` | Multi-file composer |
| Victor: Refactor | - | Refactor selected code |
| Victor: Fix | - | Fix issues in selection |
| Victor: Test | - | Generate tests |
| Victor: Document | - | Add documentation |
| Victor: Switch Model | - | Change AI model |
| Victor: Switch Mode | - | Change agent mode |
| Victor: Undo | - | Undo last change |
| Victor: Redo | - | Redo last change |
| Victor: Start Server | - | Manually start server |
| Victor: Stop Server | - | Stop server |
| Victor: Restart Server | - | Restart server |
| Victor: Server Status | - | View server status and options |
| Victor: Show Server Logs | - | View server output |

## Configuration

Open Settings (`Ctrl+,`) and search for "Victor":

```json
{
  "victor.serverUrl": "http://127.0.0.1:8765",
  "victor.serverPort": 8765,
  "victor.autoStart": true,
  "victor.profile": "default",
  "victor.mode": "build",
  "victor.autonomy.level": "semi-auto"
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `serverUrl` | Full Victor server URL for remote/shared deployments. Overrides `serverPort` when set. | `http://127.0.0.1:8765` |
| `serverPort` | Port for Victor server | `8765` |
| `autoStart` | Auto-start server on activation | `true` |
| `profile` | Victor profile from ~/.victor/profiles.yaml | `default` |
| `mode` | Agent mode (build/plan/explore) | `build` |
| `autonomy.level` | Autonomy level (manual/semi-auto/auto) | `semi-auto` |

### Available Profiles

**Local (Ollama):**
- `default` - Qwen3-Coder 30B, 64K context (recommended)
- `m4` - Qwen3-Coder 30B, 128K context
- `m4-deepseek` - DeepSeek-R1 32B with thinking
- `quick` - Mistral 7B, fastest local

**Cloud:**
- `claude` - Claude Sonnet 4.5
- `gpt` - GPT-4o
- `groq` - Llama 3.3 70B (ultra-fast)
- `gemini` - Gemini 3.0 Flash
- `deepseek` - DeepSeek-V3.2

Profiles are defined in `~/.victor/profiles.yaml`.

## Server Management

### Automatic Mode (Default)

When `victor.autoStart` is `true`:
1. Extension checks if server is already running
2. If not, spawns `victor serve` automatically
3. Waits for server to be ready
4. Shows status in status bar

### Manual Mode

Set `victor.autoStart` to `false`, then:
- Use Command Palette → "Victor: Start Server"
- Or run in terminal: `victor serve --host 127.0.0.1 --port 8765`

Some panels depend on optional server capabilities such as agents, plans, teams, workflows, or MCP. On slimmer Victor deployments those panels now stay available in the UI but show an explicit "unavailable on this server" placeholder instead of repeatedly failing.

### Status Bar

The server status bar shows:
- 🟢 **Running** - Server is healthy
- ⚪ **Stopped** - Server not running
- 🔄 **Starting** - Server is starting up
- 🔴 **Error** - Server has issues

Click to see options (start/stop/restart/logs).

## Troubleshooting

### Server won't start automatically

1. Verify Victor is installed:
   ```bash
   victor --version
   ```

2. Check Python is available:
   ```bash
   which python3
   ```

3. View server logs:
   - Command Palette → "Victor: Show Server Logs"

4. Try manual start:
   ```bash
   victor serve --port 8765 --log-level DEBUG
   ```

### Connection errors

1. Check server status in status bar
2. Verify port isn't in use: `lsof -i :8765`
3. Restart server: Command Palette → "Victor: Restart Server"

### Model errors

1. **Cloud providers**: Set API keys in `~/.victor/profiles.yaml`
2. **Ollama**: Ensure Ollama is running and model is pulled:
   ```bash
   ollama serve
   ollama pull qwen2.5-coder:7b
   ```

## Development

Use Node 24 LTS. `.nvmrc` pins the tested patch (24.21.0); the build, fast CI and
release workflows read that same file. VSCE 4 is the packaging tool. Install from
the committed lockfiles so local validation uses the same dependency graph as CI.

```bash
cd vscode-victor
nvm install
nvm use
npm ci
npm run lint
npm run compile
npm run coverage
cd webview-ui
npm ci
npm run check
npm run build
cd ..
npm test
npx --no-install vsce package --no-dependencies
```

`npm test` downloads and launches an isolated VS Code extension test host. On
headless Linux, use `xvfb-run -a npm test`. Packaging runs the production webview
and extension build through `vscode:prepublish`; do not disable it for smoke tests.

Press F5 to launch the extension in debug mode.

## License

Apache 2.0
