<script lang="ts">
  import type { ToolCall } from '../stores';

  export let toolCall: ToolCall;
  export let expanded: boolean = false;

  // Get icon based on tool category or name
  function getToolIcon(categoryOrName: string): string {
    const icons: Record<string, string> = {
      filesystem: '📁', file: '📁', read_file: '👁️', write_file: '✏️',
      search: '🔍', code_search: '🔍', semantic_code_search: '🔍',
      git: '🔀', git_tool: '🔀',
      shell: '💻', bash: '💻', execute_command: '💻',
      analysis: '📊', code_review: '📊',
      web: '🌐', web_search: '🌐', web_fetch: '🌐',
      docker: '🐳',
      testing: '🧪', test: '🧪',
      refactor: '🔧',
      lsp: '🧠', code_intelligence: '🧠',
      database: '🗃️',
      documentation: '📖',
      infrastructure: '🏗️', iac: '🏗️',
      batch: '📦',
      cache: '💾',
      mcp: '🔌',
      workflow: '⚙️',
      patch: '📝', edit: '📝',
      default: '🔧',
    };
    const key = (categoryOrName || '').toLowerCase().replace(/[^a-z_]/g, '');
    return icons[key] || icons.default;
  }

  // Calculate duration — prefer the server-measured elapsed_ms from the v1
  // wire contract; fall back to client-side start/end timestamps.
  function getDuration(): string | null {
    const ms =
      toolCall.elapsedMs ??
      (toolCall.startTime && toolCall.endTime ? toolCall.endTime - toolCall.startTime : null);
    if (ms === null || ms === undefined) return null;
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  // Toggle expanded state
  function toggle() {
    expanded = !expanded;
  }

  $: duration = getDuration();
  $: icon = getToolIcon(toolCall.category || toolCall.name);
  $: statusClass = toolCall.status;
</script>

<div
  class="tool-call"
  class:expanded
  class:dangerous={toolCall.isDangerous}
  data-status={toolCall.status}
>
  <button class="tool-call-header" on:click={toggle}>
    <span class="tool-icon">{icon}</span>
    <span class="tool-name">{toolCall.name}</span>

    <span class="tool-status">
      {#if toolCall.isDangerous}
        <span class="badge danger">⚠️</span>
      {/if}

      {#if toolCall.costTier}
        <span class="badge {toolCall.costTier}">{toolCall.costTier}</span>
      {/if}

      <span class="status-indicator {statusClass}">
        {#if toolCall.status === 'running'}
          <span class="spinner"></span>
        {/if}
      </span>

      {#if duration}
        <span class="duration">{duration}</span>
      {/if}
    </span>

    <span class="chevron">{expanded ? '▼' : '▶'}</span>
  </button>

  {#if expanded}
    <div class="tool-call-body">
      {#if Object.keys(toolCall.arguments || {}).length > 0}
        <div class="section">
          <div class="section-title">Arguments</div>
          <div class="args-container">
            {#each Object.entries(toolCall.arguments) as [key, value]}
              <div class="arg-row">
                <span class="arg-key">{key}:</span>
                <span class="arg-value">{typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      {#if toolCall.result || toolCall.error}
        <div class="section">
          <div class="section-title">Result</div>
          <div class="result-container" class:error={toolCall.status === 'error'}>
            {toolCall.error || toolCall.result}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .tool-call {
    background: var(--surface);
    border: 1px solid var(--panel-highlight);
    border-radius: var(--radius-md);
    margin: var(--spacing-sm) 0;
    overflow: hidden;
    transition: border-color var(--transition-fast);
  }

  .tool-call.dangerous {
    border-left: 3px solid var(--warning);
  }

  .tool-call[data-status="success"] {
    border-left: 3px solid var(--success);
  }

  .tool-call[data-status="error"] {
    border-left: 3px solid var(--error);
  }

  .tool-call-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    width: 100%;
    padding: var(--spacing-sm) var(--spacing-md);
    background: none;
    border: none;
    cursor: pointer;
    color: var(--text);
    font-size: var(--font-size-sm);
    text-align: left;
    transition: background-color var(--transition-fast);
  }

  .tool-call-header:hover {
    background: var(--panel);
  }

  .tool-icon {
    font-size: var(--font-size-base);
  }

  .tool-name {
    font-weight: 600;
    color: var(--accent-yellow);
    flex: 1;
    font-family: var(--vscode-editor-font-family, monospace);
  }

  .tool-status {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .badge {
    font-size: var(--font-size-xs);
    padding: 2px 6px;
    border-radius: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
  }

  .badge.free { background: var(--success); color: #000; }
  .badge.low { background: var(--primary); color: #fff; }
  .badge.medium { background: var(--warning); color: #000; }
  .badge.high { background: var(--error); color: #fff; }
  .badge.danger { background: var(--error); color: #fff; }

  .status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .status-indicator.pending { background: var(--text-muted); }
  .status-indicator.running { background: var(--primary); }
  .status-indicator.success { background: var(--success); }
  .status-indicator.error { background: var(--error); }

  .spinner {
    width: 8px;
    height: 8px;
    border: 2px solid var(--background);
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .duration {
    font-size: var(--font-size-xs);
    color: var(--text-muted);
  }

  .chevron {
    color: var(--text-subtle);
    font-size: var(--font-size-xs);
    transition: transform var(--transition-fast);
  }

  .tool-call-body {
    border-top: 1px solid var(--panel-highlight);
    padding: var(--spacing-md);
    animation: fadeIn var(--transition-normal);
  }

  .section {
    margin-bottom: var(--spacing-md);
  }

  .section:last-child {
    margin-bottom: 0;
  }

  .section-title {
    font-size: var(--font-size-xs);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: var(--spacing-xs);
  }

  .args-container {
    background: var(--panel);
    padding: var(--spacing-sm);
    border-radius: var(--radius-sm);
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: var(--font-size-sm);
    max-height: 150px;
    overflow: auto;
  }

  .arg-row {
    display: flex;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-xs);
  }

  .arg-row:last-child {
    margin-bottom: 0;
  }

  .arg-key {
    color: var(--primary-bright);
    flex-shrink: 0;
  }

  .arg-value {
    color: var(--success-bright);
    white-space: pre-wrap;
    word-break: break-all;
  }

  .result-container {
    background: var(--panel);
    padding: var(--spacing-sm);
    border-radius: var(--radius-sm);
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: var(--font-size-sm);
    max-height: 200px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    border-left: 3px solid var(--success);
  }

  .result-container.error {
    border-left-color: var(--error);
    color: var(--error-bright);
  }
</style>
