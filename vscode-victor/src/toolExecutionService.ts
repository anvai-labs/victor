/**
 * Tool Execution Service for Victor AI
 *
 * Provides real-time visibility into tool executions during chat:
 * - Tracks active and completed tool executions
 * - Emits progress events for UI updates
 * - Maintains execution history with timing metrics
 * - Supports cancellation of long-running tools
 *
 * Features:
 * - Event-driven architecture for real-time updates
 * - Execution state machine (pending → running → completed/failed/cancelled)
 * - Progress percentage tracking for batch operations
 * - WebSocket integration for live updates from backend
 */

import * as vscode from 'vscode';
import type { VictorClient } from './victorClient';

/**
 * Tool execution status
 */
export type ToolExecutionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Tool execution progress info
 */
export interface ToolExecutionProgress {
    id: string;
    toolName: string;
    displayName: string;
    status: ToolExecutionStatus;
    startTime: number;
    endTime?: number;
    duration?: number;
    progress?: number; // 0-100 percentage
    message?: string;
    result?: unknown;
    error?: string;
    arguments?: Record<string, unknown>;
}

/**
 * Tool execution events
 */
export interface ToolExecutionEvents {
    onExecutionStart: vscode.Event<ToolExecutionProgress>;
    onExecutionProgress: vscode.Event<ToolExecutionProgress>;
    onExecutionComplete: vscode.Event<ToolExecutionProgress>;
    onExecutionFailed: vscode.Event<ToolExecutionProgress>;
    onExecutionCancelled: vscode.Event<ToolExecutionProgress>;
}

/**
 * Tool metadata for display
 */
interface ToolMetadata {
    displayName: string;
    icon: string;
    category: string;
}

/**
 * Tool display metadata mapping
 */
const TOOL_METADATA: Record<string, ToolMetadata> = {
    // File system tools
    read_file: { displayName: 'Read File', icon: '$(file)', category: 'filesystem' },
    write_file: { displayName: 'Write File', icon: '$(file-add)', category: 'filesystem' },
    list_directory: { displayName: 'List Directory', icon: '$(folder)', category: 'filesystem' },
    create_directory: { displayName: 'Create Directory', icon: '$(folder-add)', category: 'filesystem' },
    delete_file: { displayName: 'Delete File', icon: '$(trash)', category: 'filesystem' },
    move_file: { displayName: 'Move File', icon: '$(move)', category: 'filesystem' },
    copy_file: { displayName: 'Copy File', icon: '$(copy)', category: 'filesystem' },

    // Code search tools
    code_search: { displayName: 'Code Search', icon: '$(search)', category: 'search' },
    semantic_code_search: { displayName: 'Semantic Search', icon: '$(symbol-keyword)', category: 'search' },
    grep_search: { displayName: 'Grep Search', icon: '$(regex)', category: 'search' },
    find_files: { displayName: 'Find Files', icon: '$(search)', category: 'search' },

    // Git tools
    git_status: { displayName: 'Git Status', icon: '$(git-branch)', category: 'git' },
    git_diff: { displayName: 'Git Diff', icon: '$(diff)', category: 'git' },
    git_commit: { displayName: 'Git Commit', icon: '$(git-commit)', category: 'git' },
    git_log: { displayName: 'Git Log', icon: '$(history)', category: 'git' },
    git_branch: { displayName: 'Git Branch', icon: '$(git-branch)', category: 'git' },

    // Code analysis tools
    code_review: { displayName: 'Code Review', icon: '$(eye)', category: 'analysis' },
    refactor: { displayName: 'Refactor', icon: '$(edit)', category: 'analysis' },
    analyze_code: { displayName: 'Analyze Code', icon: '$(microscope)', category: 'analysis' },
    lint: { displayName: 'Lint', icon: '$(checklist)', category: 'analysis' },

    // Testing tools
    run_tests: { displayName: 'Run Tests', icon: '$(beaker)', category: 'testing' },
    generate_tests: { displayName: 'Generate Tests', icon: '$(beaker)', category: 'testing' },
    test_coverage: { displayName: 'Test Coverage', icon: '$(graph)', category: 'testing' },

    // Documentation tools
    generate_docs: { displayName: 'Generate Docs', icon: '$(book)', category: 'docs' },
    update_readme: { displayName: 'Update README', icon: '$(markdown)', category: 'docs' },

    // Shell/bash tools
    bash: { displayName: 'Run Command', icon: '$(terminal)', category: 'shell' },
    execute_command: { displayName: 'Execute Command', icon: '$(terminal)', category: 'shell' },

    // Web tools
    web_search: { displayName: 'Web Search', icon: '$(globe)', category: 'web' },
    web_fetch: { displayName: 'Fetch URL', icon: '$(link)', category: 'web' },

    // Docker tools
    docker_build: { displayName: 'Docker Build', icon: '$(package)', category: 'docker' },
    docker_run: { displayName: 'Docker Run', icon: '$(play)', category: 'docker' },
    docker_ps: { displayName: 'Docker PS', icon: '$(list-flat)', category: 'docker' },

    // Database tools
    query_database: { displayName: 'Query Database', icon: '$(database)', category: 'database' },

    // LSP tools
    lsp_hover: { displayName: 'Get Hover', icon: '$(info)', category: 'lsp' },
    lsp_definition: { displayName: 'Go to Definition', icon: '$(references)', category: 'lsp' },
    lsp_references: { displayName: 'Find References', icon: '$(references)', category: 'lsp' },
    lsp_completions: { displayName: 'Get Completions', icon: '$(symbol-method)', category: 'lsp' },

    // MCP tools
    mcp_call: { displayName: 'MCP Call', icon: '$(plug)', category: 'mcp' },
};

/**
 * Default metadata for unknown tools
 */
const DEFAULT_TOOL_METADATA: ToolMetadata = {
    displayName: 'Tool',
    icon: '$(tools)',
    category: 'other',
};

/**
 * Tool Execution Service
 */
export class ToolExecutionService implements vscode.Disposable {
    private executions: Map<string, ToolExecutionProgress> = new Map();
    private executionHistory: ToolExecutionProgress[] = [];
    private readonly maxHistorySize = 100;

    // Event emitters
    private _onExecutionStart = new vscode.EventEmitter<ToolExecutionProgress>();
    private _onExecutionProgress = new vscode.EventEmitter<ToolExecutionProgress>();
    private _onExecutionComplete = new vscode.EventEmitter<ToolExecutionProgress>();
    private _onExecutionFailed = new vscode.EventEmitter<ToolExecutionProgress>();
    private _onExecutionCancelled = new vscode.EventEmitter<ToolExecutionProgress>();

    // Public events
    readonly onExecutionStart = this._onExecutionStart.event;
    readonly onExecutionProgress = this._onExecutionProgress.event;
    readonly onExecutionComplete = this._onExecutionComplete.event;
    readonly onExecutionFailed = this._onExecutionFailed.event;
    readonly onExecutionCancelled = this._onExecutionCancelled.event;

    private victorClient: VictorClient | null = null;
    private statusBarItem: vscode.StatusBarItem;

    constructor() {
        // Create status bar item for tool execution visibility
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            90
        );
        this.statusBarItem.command = 'victor.showToolExecutions';
        this.updateStatusBar();
    }

    /**
     * Set the Victor client for WebSocket integration
     */
    setVictorClient(client: VictorClient): void {
        this.victorClient = client;
    }

    /**
     * Start tracking a tool execution
     */
    startExecution(
        id: string,
        toolName: string,
        args?: Record<string, unknown>
    ): ToolExecutionProgress {
        const metadata = TOOL_METADATA[toolName] || {
            ...DEFAULT_TOOL_METADATA,
            displayName: this.formatToolName(toolName),
        };

        const execution: ToolExecutionProgress = {
            id,
            toolName,
            displayName: metadata.displayName,
            status: 'running',
            startTime: Date.now(),
            arguments: args,
        };

        this.executions.set(id, execution);
        this._onExecutionStart.fire(execution);
        this.updateStatusBar();

        return execution;
    }

    /**
     * Update execution progress
     */
    updateProgress(id: string, progress: number, message?: string): void {
        const execution = this.executions.get(id);
        if (!execution) {
            return;
        }

        execution.progress = Math.min(100, Math.max(0, progress));
        if (message) {
            execution.message = message;
        }

        this._onExecutionProgress.fire(execution);
        this.updateStatusBar();
    }

    /**
     * Mark execution as completed
     */
    completeExecution(id: string, result?: unknown): void {
        const execution = this.executions.get(id);
        if (!execution) {
            return;
        }

        execution.status = 'completed';
        execution.endTime = Date.now();
        execution.duration = execution.endTime - execution.startTime;
        execution.progress = 100;
        execution.result = result;

        this.addToHistory(execution);
        this.executions.delete(id);
        this._onExecutionComplete.fire(execution);
        this.updateStatusBar();
    }

    /**
     * Mark execution as failed
     */
    failExecution(id: string, error: string): void {
        const execution = this.executions.get(id);
        if (!execution) {
            return;
        }

        execution.status = 'failed';
        execution.endTime = Date.now();
        execution.duration = execution.endTime - execution.startTime;
        execution.error = error;

        this.addToHistory(execution);
        this.executions.delete(id);
        this._onExecutionFailed.fire(execution);
        this.updateStatusBar();
    }

    /**
     * Cancel an execution
     */
    async cancelExecution(id: string): Promise<boolean> {
        const execution = this.executions.get(id);
        if (!execution) {
            return false;
        }

        // Only a confirmed pending-approval cancellation can be shown as
        // cancelled. Missing transport/negative acknowledgement leaves it active.
        if (!this.victorClient) return false;
        try {
            if (!await this.victorClient.cancelToolExecution(id)) return false;
        } catch {
            return false;
        }

        // Completion/failure or a replacement may have arrived while awaiting HTTP.
        if (this.executions.get(id) !== execution) return false;

        execution.status = 'cancelled';
        execution.endTime = Date.now();
        execution.duration = execution.endTime - execution.startTime;

        this.addToHistory(execution);
        this.executions.delete(id);
        this._onExecutionCancelled.fire(execution);
        this.updateStatusBar();

        return true;
    }

    /**
     * Get all active executions
     */
    getActiveExecutions(): ToolExecutionProgress[] {
        return Array.from(this.executions.values());
    }

    /**
     * Get execution history
     */
    getHistory(): ToolExecutionProgress[] {
        return [...this.executionHistory];
    }

    /**
     * Get execution by ID
     */
    getExecution(id: string): ToolExecutionProgress | undefined {
        return this.executions.get(id) || this.executionHistory.find(e => e.id === id);
    }

    /**
     * Clear execution history
     */
    clearHistory(): void {
        this.executionHistory = [];
    }

    /**
     * Handle tool execution event from WebSocket
     */
    handleToolEvent(event: {
        type: 'start' | 'progress' | 'complete' | 'error';
        tool_call_id: string;
        tool_name: string;
        progress?: number;
        message?: string;
        result?: unknown;
        error?: string;
        arguments?: Record<string, unknown>;
    }): void {
        switch (event.type) {
            case 'start':
                this.startExecution(event.tool_call_id, event.tool_name, event.arguments);
                break;
            case 'progress':
                this.updateProgress(event.tool_call_id, event.progress || 0, event.message);
                break;
            case 'complete':
                this.completeExecution(event.tool_call_id, event.result);
                break;
            case 'error':
                this.failExecution(event.tool_call_id, event.error || 'Unknown error');
                break;
        }
    }

    /**
     * Format tool name for display
     */
    private formatToolName(name: string): string {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Add execution to history with size limit
     */
    private addToHistory(execution: ToolExecutionProgress): void {
        this.executionHistory.unshift(execution);
        if (this.executionHistory.length > this.maxHistorySize) {
            this.executionHistory.pop();
        }
    }

    /**
     * Update status bar with active executions
     */
    private updateStatusBar(): void {
        const activeCount = this.executions.size;

        if (activeCount === 0) {
            this.statusBarItem.hide();
            return;
        }

        const executions = Array.from(this.executions.values());
        const currentTool = executions[0];

        if (activeCount === 1) {
            const metadata = TOOL_METADATA[currentTool.toolName] || DEFAULT_TOOL_METADATA;
            const progressStr = currentTool.progress !== undefined
                ? ` (${Math.round(currentTool.progress)}%)`
                : '';
            this.statusBarItem.text = `${metadata.icon} ${currentTool.displayName}${progressStr}`;
            this.statusBarItem.tooltip = currentTool.message || `Running ${currentTool.displayName}...`;
        } else {
            this.statusBarItem.text = `$(tools) ${activeCount} tools running`;
            this.statusBarItem.tooltip = executions.map(e => `- ${e.displayName}`).join('\n');
        }

        this.statusBarItem.show();
    }

    /**
     * Dispose resources
     */
    dispose(): void {
        this.statusBarItem.dispose();
        this._onExecutionStart.dispose();
        this._onExecutionProgress.dispose();
        this._onExecutionComplete.dispose();
        this._onExecutionFailed.dispose();
        this._onExecutionCancelled.dispose();
        this.executions.clear();
        this.executionHistory = [];
    }
}

/**
 * Register tool execution commands
 */
export function registerToolExecutionCommands(
    context: vscode.ExtensionContext,
    service: ToolExecutionService
): void {
    // Show tool executions
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showToolExecutions', async () => {
            const active = service.getActiveExecutions();
            const history = service.getHistory().slice(0, 10);

            if (active.length === 0 && history.length === 0) {
                vscode.window.showInformationMessage('No tool executions to display.');
                return;
            }

            const items: vscode.QuickPickItem[] = [];

            if (active.length > 0) {
                items.push({
                    label: '$(sync~spin) Active Executions',
                    kind: vscode.QuickPickItemKind.Separator,
                });

                for (const exec of active) {
                    const duration = Math.round((Date.now() - exec.startTime) / 1000);
                    const progressStr = exec.progress !== undefined
                        ? ` (${Math.round(exec.progress)}%)`
                        : '';
                    items.push({
                        label: `$(tools) ${exec.displayName}${progressStr}`,
                        description: `Running for ${duration}s`,
                        detail: exec.message || `ID: ${exec.id}`,
                    });
                }
            }

            if (history.length > 0) {
                items.push({
                    label: '$(history) Recent History',
                    kind: vscode.QuickPickItemKind.Separator,
                });

                for (const exec of history) {
                    const icon = exec.status === 'completed' ? '$(check)' :
                                 exec.status === 'failed' ? '$(error)' :
                                 exec.status === 'cancelled' ? '$(close)' : '$(question)';
                    const duration = exec.duration !== undefined
                        ? `${Math.round(exec.duration)}ms`
                        : 'unknown';
                    items.push({
                        label: `${icon} ${exec.displayName}`,
                        description: duration,
                        detail: exec.error || exec.message || `Status: ${exec.status}`,
                    });
                }
            }

            await vscode.window.showQuickPick(items, {
                placeHolder: 'Tool Executions',
                canPickMany: false,
            });
        })
    );

    // Cancel all active executions
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.cancelAllToolExecutions', async () => {
            const active = service.getActiveExecutions();
            if (active.length === 0) {
                vscode.window.showInformationMessage('No active tool executions to cancel.');
                return;
            }

            const confirm = await vscode.window.showWarningMessage(
                `Cancel ${active.length} active tool execution(s)?`,
                { modal: true },
                'Cancel All'
            );

            if (confirm === 'Cancel All') {
                let cancelled = 0;
                for (const exec of active) {
                    if (await service.cancelExecution(exec.id)) cancelled++;
                }
                vscode.window.showInformationMessage(
                    `Cancellation confirmed for ${cancelled} of ${active.length} tool executions. ` +
                    'Unconfirmed executions may still be running.'
                );
            }
        })
    );

    // Clear execution history
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.clearToolExecutionHistory', () => {
            service.clearHistory();
            vscode.window.showInformationMessage('Tool execution history cleared.');
        })
    );
}
