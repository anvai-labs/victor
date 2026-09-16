/**
 * Terminal Provider
 *
 * Provides terminal integration for AI-executed commands:
 * - Dedicated Victor terminal for command execution
 * - Command history tracking
 * - Output capture and streaming
 * - Interactive command approval
 */

import * as vscode from 'vscode';
import * as cp from 'child_process';

export interface CommandExecution {
    id: string;
    command: string;
    cwd: string;
    startTime: Date;
    endTime?: Date;
    exitCode?: number;
    output: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
}

export type CommandApprovalCallback = (command: string, cwd: string) => Promise<boolean>;

/**
 * Manages Victor's dedicated terminal for command execution
 */
export class TerminalProvider implements vscode.Disposable {
    private _terminal: vscode.Terminal | null = null;
    private _executions: CommandExecution[] = [];
    private _maxHistory = 100;
    private _outputChannel: vscode.OutputChannel;
    private _approvalCallback?: CommandApprovalCallback;
    private _writeEmitter = new vscode.EventEmitter<string>();
    private _disposables: vscode.Disposable[] = [];

    // Dangerous command patterns that require approval
    private readonly _dangerousPatterns = [
        /\brm\s+(-rf?|--recursive)/i,
        /\bgit\s+(push|reset|rebase|force)/i,
        /\bsudo\b/i,
        /\bchmod\s+777/i,
        /\bdel\s+\/[sfq]/i,  // Windows delete
        /\bformat\b/i,
        /\bmkfs\b/i,
        /\bdd\s+if=/i,
        />\s*\/dev\//i,
        /\bcurl\b.*\|\s*(ba)?sh/i,  // Piping curl to shell
        /\bwget\b.*\|\s*(ba)?sh/i,
    ];

    constructor() {
        this._outputChannel = vscode.window.createOutputChannel('Victor Commands');

        // Listen for terminal close
        this._disposables.push(
            vscode.window.onDidCloseTerminal((terminal) => {
                if (terminal === this._terminal) {
                    this._terminal = null;
                }
            })
        );
    }

    /**
     * Set callback for command approval
     */
    setApprovalCallback(callback: CommandApprovalCallback): void {
        this._approvalCallback = callback;
    }

    /**
     * Get or create the Victor terminal
     */
    getTerminal(cwd?: string): vscode.Terminal {
        if (!this._terminal || cwd !== undefined) {
            this._terminal = vscode.window.createTerminal({
                name: 'Victor AI',
                cwd: cwd || undefined,
                iconPath: new vscode.ThemeIcon('hubot'),
            });
            this._disposables.push(this._terminal);
        }
        return this._terminal;
    }

    /**
     * Execute a command in the terminal
     */
    async executeCommand(
        command: string,
        options: {
            cwd?: string;
            requireApproval?: boolean;
            showTerminal?: boolean;
        } = {}
    ): Promise<CommandExecution> {
        const {
            cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '',
            requireApproval = true,
            showTerminal = true,
        } = options;

        const execution: CommandExecution = {
            id: `exec-${Date.now()}`,
            command,
            cwd,
            startTime: new Date(),
            output: '',
            status: 'pending',
        };

        this._executions.unshift(execution);
        if (this._executions.length > this._maxHistory) {
            this._executions.pop();
        }

        // Check for dangerous commands
        const isDangerous = this._isDangerousCommand(command);

        // Request approval if needed
        if (requireApproval && (isDangerous || this._approvalCallback)) {
            const approved = await this._requestApproval(command, cwd, isDangerous);
            if (!approved) {
                execution.status = 'cancelled';
                execution.endTime = new Date();
                return execution;
            }
        }

        // Execute the command
        execution.status = 'running';
        // Start in the requested directory through VS Code, without shell text.
        // A fresh command terminal also avoids inheriting an interactive cd.
        const terminal = this.getTerminal(cwd);

        if (showTerminal) {
            terminal.show(true);
        }

        // Log to output channel
        this._outputChannel.appendLine(`\n[${new Date().toLocaleTimeString()}] Executing: ${command}`);
        this._outputChannel.appendLine(`  Working directory: ${cwd}`);

        // Send command
        terminal.sendText(command, true);

        // Mark as completed (we can't easily get exit code from terminal API)
        execution.status = 'completed';
        execution.endTime = new Date();

        return execution;
    }

    /**
     * Execute command and capture output (using child_process)
     */
    async executeWithOutput(
        command: string,
        options: {
            cwd?: string;
            timeout?: number;
            requireApproval?: boolean;
        } = {}
    ): Promise<CommandExecution> {
        const {
            cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '',
            timeout = 60000,
            requireApproval = true,
        } = options;

        const execution: CommandExecution = {
            id: `exec-${Date.now()}`,
            command,
            cwd,
            startTime: new Date(),
            output: '',
            status: 'pending',
        };

        this._executions.unshift(execution);

        // Check for dangerous commands
        const isDangerous = this._isDangerousCommand(command);

        if (requireApproval && (isDangerous || this._approvalCallback)) {
            const approved = await this._requestApproval(command, cwd, isDangerous);
            if (!approved) {
                execution.status = 'cancelled';
                execution.endTime = new Date();
                return execution;
            }
        }

        execution.status = 'running';

        return new Promise((resolve) => {
            const child = cp.spawn(command, [], {
                shell: true,
                cwd,
                timeout,
            });

            let stdout = '';
            let stderr = '';

            child.stdout?.on('data', (data: Buffer) => {
                const text = data.toString();
                stdout += text;
                this._outputChannel.append(text);
            });

            child.stderr?.on('data', (data: Buffer) => {
                const text = data.toString();
                stderr += text;
                this._outputChannel.append(text);
            });

            child.on('close', (code: number | null) => {
                execution.exitCode = code ?? undefined;
                execution.output = stdout + (stderr ? `\nSTDERR:\n${stderr}` : '');
                execution.status = code === 0 ? 'completed' : 'failed';
                execution.endTime = new Date();
                resolve(execution);
            });

            child.on('error', (err: Error) => {
                execution.output = `Error: ${err.message}`;
                execution.status = 'failed';
                execution.endTime = new Date();
                resolve(execution);
            });
        });
    }

    /**
     * Check if command is potentially dangerous
     */
    private _isDangerousCommand(command: string): boolean {
        return this._dangerousPatterns.some(pattern => pattern.test(command));
    }

    /**
     * Request user approval for command execution
     */
    private async _requestApproval(
        command: string,
        cwd: string,
        isDangerous: boolean
    ): Promise<boolean> {
        // Use custom callback if provided
        if (this._approvalCallback) {
            return this._approvalCallback(command, cwd);
        }

        // Show approval dialog
        const dangerWarning = isDangerous
            ? '\n\n⚠️ This command may be dangerous!'
            : '';

        const result = await vscode.window.showWarningMessage(
            `Victor wants to run a command:${dangerWarning}`,
            {
                modal: true,
                detail: `Command: ${command}\nDirectory: ${cwd}`,
            },
            'Run',
            'Run All',
            'Cancel'
        );

        if (result === 'Run All') {
            // Disable approval for this session
            this._approvalCallback = async () => true;
        }

        return result === 'Run' || result === 'Run All';
    }

    /**
     * Get command execution history
     */
    getHistory(count?: number): CommandExecution[] {
        return this._executions.slice(0, count || this._maxHistory);
    }

    /**
     * Get last execution
     */
    getLastExecution(): CommandExecution | undefined {
        return this._executions[0];
    }

    /**
     * Clear history
     */
    clearHistory(): void {
        this._executions = [];
    }

    /**
     * Show output channel
     */
    showOutput(): void {
        this._outputChannel.show();
    }

    /**
     * Show terminal
     */
    showTerminal(): void {
        this.getTerminal().show();
    }

    /**
     * Kill current terminal process (if possible)
     */
    killTerminal(): void {
        if (this._terminal) {
            this._terminal.dispose();
            this._terminal = null;
        }
    }

    dispose(): void {
        if (this._terminal) {
            this._terminal.dispose();
        }
        this._outputChannel.dispose();
        this._writeEmitter.dispose();
        for (const d of this._disposables) {
            d.dispose();
        }
    }
}

/**
 * Terminal history view provider
 */
export class TerminalHistoryProvider implements vscode.TreeDataProvider<CommandItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<CommandItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    constructor(private readonly _terminalProvider: TerminalProvider) {}

    refresh(): void {
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: CommandItem): vscode.TreeItem {
        return element;
    }

    getChildren(): CommandItem[] {
        const history = this._terminalProvider.getHistory(20);

        if (history.length === 0) {
            return [new CommandItem({
                id: 'empty',
                command: 'No commands executed yet',
                cwd: '',
                startTime: new Date(),
                output: '',
                status: 'pending',
            })];
        }

        return history.map(exec => new CommandItem(exec));
    }
}

class CommandItem extends vscode.TreeItem {
    constructor(public readonly execution: CommandExecution) {
        super(execution.command, vscode.TreeItemCollapsibleState.None);

        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = execution.status;

        if (execution.output) {
            this.command = {
                command: 'victor.showCommandOutput',
                title: 'Show Output',
                arguments: [execution],
            };
        }
    }

    private _getDescription(): string {
        const duration = this.execution.endTime
            ? `${((this.execution.endTime.getTime() - this.execution.startTime.getTime()) / 1000).toFixed(1)}s`
            : 'running...';

        return `${this.execution.status} (${duration})`;
    }

    private _getTooltip(): string {
        return [
            `Command: ${this.execution.command}`,
            `Directory: ${this.execution.cwd}`,
            `Status: ${this.execution.status}`,
            this.execution.exitCode !== undefined ? `Exit code: ${this.execution.exitCode}` : '',
            `Started: ${this.execution.startTime.toLocaleTimeString()}`,
        ].filter(Boolean).join('\n');
    }

    private _getIcon(): vscode.ThemeIcon {
        switch (this.execution.status) {
            case 'running':
                return new vscode.ThemeIcon('sync~spin');
            case 'completed':
                return new vscode.ThemeIcon('check', new vscode.ThemeColor('testing.iconPassed'));
            case 'failed':
                return new vscode.ThemeIcon('error', new vscode.ThemeColor('testing.iconFailed'));
            case 'cancelled':
                return new vscode.ThemeIcon('circle-slash');
            default:
                return new vscode.ThemeIcon('terminal');
        }
    }
}

/**
 * Register terminal commands
 */
export function registerTerminalCommands(
    context: vscode.ExtensionContext,
    terminalProvider: TerminalProvider
): void {
    // Show Victor terminal
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showTerminal', () => {
            terminalProvider.showTerminal();
        })
    );

    // Show command output
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showCommandOutput', (execution: CommandExecution) => {
            const output = vscode.window.createOutputChannel(`Victor: ${execution.command}`);
            output.clear();
            output.appendLine(`Command: ${execution.command}`);
            output.appendLine(`Directory: ${execution.cwd}`);
            output.appendLine(`Status: ${execution.status}`);
            if (execution.exitCode !== undefined) {
                output.appendLine(`Exit code: ${execution.exitCode}`);
            }
            output.appendLine(`\n--- Output ---\n`);
            output.appendLine(execution.output || '(no output)');
            output.show();
        })
    );

    // Show command logs
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showCommandLogs', () => {
            terminalProvider.showOutput();
        })
    );

    // Kill terminal
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.killTerminal', () => {
            terminalProvider.killTerminal();
            vscode.window.showInformationMessage('Victor terminal killed');
        })
    );

    // Run command (for testing)
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.runCommand', async () => {
            const command = await vscode.window.showInputBox({
                prompt: 'Enter command to execute',
                placeHolder: 'e.g., npm test',
            });

            if (command) {
                const execution = await terminalProvider.executeCommand(command);
                if (execution.status === 'cancelled') {
                    vscode.window.showWarningMessage('Command cancelled');
                }
            }
        })
    );

    // Clear command history
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.clearCommandHistory', () => {
            terminalProvider.clearHistory();
            vscode.window.showInformationMessage('Command history cleared');
        })
    );
}
