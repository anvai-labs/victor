/**
 * Terminal Agent Provider
 *
 * Provides AI-assisted terminal command execution with approval workflow.
 * Similar to Cursor's terminal agent and Copilot's command suggestions.
 */

import * as vscode from 'vscode';
import { VictorClient, requireCompletedChat, ChatOutcomeError } from './victorClient';
import { TerminalHistoryService } from './terminalHistory';

export interface TerminalCommand {
    id: string;
    command: string;
    description: string;
    workingDir: string;
    status: 'pending' | 'approved' | 'rejected' | 'running' | 'completed' | 'failed';
    output?: string;
    exitCode?: number;
    timestamp: number;
    isDangerous: boolean;
}

export interface TerminalSession {
    id: string;
    name: string;
    terminal: vscode.Terminal;
    commands: TerminalCommand[];
    isActive: boolean;
}

/**
 * Tree item for terminal agent view
 */
class TerminalTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly itemType: 'session' | 'command',
        public readonly data?: TerminalSession | TerminalCommand
    ) {
        super(label, collapsibleState);
        this.contextValue = itemType;

        if (itemType === 'session' && data) {
            const session = data as TerminalSession;
            this.iconPath = new vscode.ThemeIcon(session.isActive ? 'terminal' : 'terminal-view-icon');
            this.description = session.isActive ? 'Active' : '';
        } else if (itemType === 'command' && data) {
            const cmd = data as TerminalCommand;
            this.description = cmd.status;
            this.tooltip = `${cmd.command}\n\n${cmd.description}`;

            // Set icon based on status
            switch (cmd.status) {
                case 'pending':
                    this.iconPath = new vscode.ThemeIcon('question', new vscode.ThemeColor('editorWarning.foreground'));
                    break;
                case 'approved':
                    this.iconPath = new vscode.ThemeIcon('check', new vscode.ThemeColor('testing.iconPassed'));
                    break;
                case 'rejected':
                    this.iconPath = new vscode.ThemeIcon('x', new vscode.ThemeColor('errorForeground'));
                    break;
                case 'running':
                    this.iconPath = new vscode.ThemeIcon('loading~spin');
                    break;
                case 'completed':
                    this.iconPath = new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'));
                    break;
                case 'failed':
                    this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
                    break;
            }

            // Mark dangerous commands
            if (cmd.isDangerous) {
                this.description = `⚠️ ${cmd.status}`;
            }
        }
    }
}

/**
 * Terminal Agent View Provider
 *
 * Manages the terminal agent sidebar panel.
 */
export class TerminalAgentProvider implements vscode.TreeDataProvider<TerminalTreeItem>, vscode.Disposable {
    private _onDidChangeTreeData = new vscode.EventEmitter<TerminalTreeItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private _sessions: Map<string, TerminalSession> = new Map();
    private _disposables: vscode.Disposable[] = [];
    private _pendingCommands: Map<string, TerminalCommand> = new Map();

    constructor(
        private readonly _client: VictorClient,
        private readonly _log?: vscode.OutputChannel
    ) {
        // Listen for terminal events
        this._disposables.push(
            vscode.window.onDidOpenTerminal(terminal => {
                this._onTerminalOpened(terminal);
            }),
            vscode.window.onDidCloseTerminal(terminal => {
                this._onTerminalClosed(terminal);
            }),
            vscode.window.onDidChangeActiveTerminal(terminal => {
                this._onActiveTerminalChanged(terminal);
            })
        );

        // Register existing terminals
        vscode.window.terminals.forEach(terminal => {
            this._onTerminalOpened(terminal);
        });
    }

    dispose(): void {
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
        this._sessions.clear();
        this._pendingCommands.clear();
    }

    getTreeItem(element: TerminalTreeItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: TerminalTreeItem): TerminalTreeItem[] {
        if (!element) {
            // Root level: show sessions
            const items: TerminalTreeItem[] = [];

            // Show pending commands first
            if (this._pendingCommands.size > 0) {
                items.push(new TerminalTreeItem(
                    `Pending Approval (${this._pendingCommands.size})`,
                    vscode.TreeItemCollapsibleState.Expanded,
                    'session'
                ));
            }

            // Show terminal sessions
            for (const session of this._sessions.values()) {
                items.push(new TerminalTreeItem(
                    session.name,
                    vscode.TreeItemCollapsibleState.Collapsed,
                    'session',
                    session
                ));
            }

            if (items.length === 0) {
                items.push(new TerminalTreeItem(
                    'No terminals open',
                    vscode.TreeItemCollapsibleState.None,
                    'session'
                ));
            }

            return items;
        }

        // Child level: show commands in session
        if (element.itemType === 'session' && element.data) {
            const session = element.data as TerminalSession;
            return session.commands.map(cmd =>
                new TerminalTreeItem(
                    cmd.command.length > 50 ? cmd.command.slice(0, 50) + '...' : cmd.command,
                    vscode.TreeItemCollapsibleState.None,
                    'command',
                    cmd
                )
            );
        }

        // Show pending commands
        if (element.label?.startsWith('Pending')) {
            return Array.from(this._pendingCommands.values()).map(cmd =>
                new TerminalTreeItem(
                    cmd.command.length > 50 ? cmd.command.slice(0, 50) + '...' : cmd.command,
                    vscode.TreeItemCollapsibleState.None,
                    'command',
                    cmd
                )
            );
        }

        return [];
    }

    /**
     * Suggest a command based on user intent
     */
    async suggestCommand(intent: string): Promise<TerminalCommand | null> {
        try {
            this._log?.appendLine(`[Terminal] Suggesting command for: ${intent}`);

            // Get workspace context
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            const workingDir = workspaceFolder?.uri.fsPath || process.cwd();

            // Call Victor API to generate command
            const response = await this._client.chat([
                {
                    role: 'user',
                    content: `Generate a terminal command for this task. Return ONLY the command, nothing else.

Working directory: ${workingDir}
OS: ${process.platform}
Task: ${intent}

Respond with just the command to run.`
                }
            ]);

            const command = requireCompletedChat(response).content?.trim() || '';
            if (!command) {
                return null;
            }

            // Clean up command (remove markdown code blocks if present)
            const cleanCommand = command
                .replace(/^```\w*\n?/gm, '')
                .replace(/\n?```$/gm, '')
                .trim();

            const cmd: TerminalCommand = {
                id: `cmd-${Date.now()}`,
                command: cleanCommand,
                description: intent,
                workingDir,
                status: 'pending',
                timestamp: Date.now(),
                isDangerous: this._isDangerousCommand(cleanCommand),
            };

            this._pendingCommands.set(cmd.id, cmd);
            this._refresh();

            // Show approval dialog for dangerous commands with explanation
            if (cmd.isDangerous) {
                const dangerReason = this._getDangerReason(cleanCommand);
                const action = await vscode.window.showWarningMessage(
                    `⚠️ Potentially dangerous command:\n${cleanCommand}\n\nReason: ${dangerReason}`,
                    { modal: true },
                    'Run Anyway',
                    'Reject'
                );

                if (action === 'Run Anyway') {
                    await this.approveCommand(cmd.id);
                } else {
                    await this.rejectCommand(cmd.id);
                }
            }

            return cmd;
        } catch (error) {
            if (error instanceof ChatOutcomeError) { void vscode.window.showWarningMessage(error.message); }
            this._log?.appendLine(`[Terminal] Error suggesting command: ${error}`);
            return null;
        }
    }

    /**
     * Approve a pending command for execution
     */
    async approveCommand(commandId: string): Promise<void> {
        const cmd = this._pendingCommands.get(commandId);
        if (!cmd) {
            return;
        }

        cmd.status = 'approved';
        this._refresh();

        // Execute the command
        await this.executeCommand(cmd);
    }

    /**
     * Reject a pending command
     */
    async rejectCommand(commandId: string): Promise<void> {
        const cmd = this._pendingCommands.get(commandId);
        if (!cmd) {
            return;
        }

        cmd.status = 'rejected';
        this._pendingCommands.delete(commandId);
        this._refresh();

        this._log?.appendLine(`[Terminal] Command rejected: ${cmd.command}`);
    }

    /**
     * Execute a command in the terminal
     */
    async executeCommand(cmd: TerminalCommand): Promise<void> {
        cmd.status = 'running';
        this._pendingCommands.delete(cmd.id);
        this._refresh();

        // Get or create terminal
        let terminal = vscode.window.activeTerminal;
        if (!terminal) {
            terminal = vscode.window.createTerminal({
                name: 'Victor Agent',
                cwd: cmd.workingDir,
            });
        }

        // Find session and add command
        const sessionId = this._getTerminalId(terminal);
        const session = this._sessions.get(sessionId);
        if (session) {
            session.commands.push(cmd);
        }

        // Show terminal and execute
        terminal.show();
        terminal.sendText(cmd.command);

        this._log?.appendLine(`[Terminal] Executing: ${cmd.command}`);

        // Add to shared terminal history service
        const historyService = TerminalHistoryService.getInstance();
        historyService.addCommand({
            command: cmd.command,
            terminalName: terminal.name,
            workingDir: cmd.workingDir,
        });

        // Mark as completed (we can't track actual completion from VS Code API)
        setTimeout(() => {
            cmd.status = 'completed';
            this._refresh();
        }, 1000);
    }

    /**
     * Run a command directly (for non-AI initiated commands)
     */
    async runCommand(command: string, description?: string): Promise<void> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        const workingDir = workspaceFolder?.uri.fsPath || process.cwd();

        const cmd: TerminalCommand = {
            id: `cmd-${Date.now()}`,
            command,
            description: description || 'Manual command',
            workingDir,
            status: 'pending',
            timestamp: Date.now(),
            isDangerous: this._isDangerousCommand(command),
        };

        if (cmd.isDangerous) {
            const dangerReason = this._getDangerReason(command);
            const action = await vscode.window.showWarningMessage(
                `⚠️ This command may be dangerous:\n${command}\n\nReason: ${dangerReason}`,
                'Run',
                'Cancel'
            );

            if (action !== 'Run') {
                return;
            }
        }

        cmd.status = 'approved';
        await this.executeCommand(cmd);
    }

    /**
     * Show command history for current session
     */
    getCommandHistory(): TerminalCommand[] {
        const terminal = vscode.window.activeTerminal;
        if (!terminal) {
            return [];
        }

        const sessionId = this._getTerminalId(terminal);
        const session = this._sessions.get(sessionId);
        return session?.commands || [];
    }

    // --- Private Methods ---

    private _onTerminalOpened(terminal: vscode.Terminal): void {
        const id = this._getTerminalId(terminal);

        if (!this._sessions.has(id)) {
            this._sessions.set(id, {
                id,
                name: terminal.name,
                terminal,
                commands: [],
                isActive: vscode.window.activeTerminal === terminal,
            });

            this._log?.appendLine(`[Terminal] Session opened: ${terminal.name}`);
            this._refresh();
        }
    }

    private _onTerminalClosed(terminal: vscode.Terminal): void {
        const id = this._getTerminalId(terminal);
        this._sessions.delete(id);
        this._log?.appendLine(`[Terminal] Session closed: ${terminal.name}`);
        this._refresh();
    }

    private _onActiveTerminalChanged(terminal: vscode.Terminal | undefined): void {
        for (const session of this._sessions.values()) {
            session.isActive = terminal ? session.terminal === terminal : false;
        }
        this._refresh();
    }

    private _getTerminalId(terminal: vscode.Terminal): string {
        // Use terminal name + processId as unique identifier
        return `terminal-${terminal.name}-${terminal.processId || 'unknown'}`;
    }

    private _isDangerousCommand(command: string): boolean {
        return this._getDangerReason(command) !== null;
    }

    /**
     * Get the reason why a command is considered dangerous
     */
    private _getDangerReason(command: string): string | null {
        // Built-in dangerous patterns with explanations
        const builtInPatterns: Array<{ pattern: RegExp; reason: string }> = [
            { pattern: /\brm\s+-rf?\s+[/~]/i, reason: 'Recursive file deletion from root or home directory' },
            { pattern: /\brm\s+.*\*/, reason: 'File deletion with wildcards' },
            { pattern: /\bsudo\s+rm\b/i, reason: 'Privileged file deletion' },
            { pattern: /\bmkfs\b/i, reason: 'Disk formatting command' },
            { pattern: /\bdd\s+if=/i, reason: 'Low-level disk write (dd)' },
            { pattern: /\b:\s*\(\)\s*\{\s*:\s*\|\s*:/, reason: 'Fork bomb pattern detected' },
            { pattern: /\bchmod\s+-R\s+777/i, reason: 'Recursive permission change to world-writable' },
            { pattern: /\bchown\s+-R\s+root/i, reason: 'Recursive ownership change to root' },
            { pattern: />\s*\/dev\/sd[a-z]/i, reason: 'Direct write to disk device' },
            { pattern: /\bcurl\s+.*\|\s*sh\b/i, reason: 'Remote code execution (curl | sh)' },
            { pattern: /\bwget\s+.*\|\s*sh\b/i, reason: 'Remote code execution (wget | sh)' },
            { pattern: /\beval\s+.*\$\(/, reason: 'Eval with command substitution' },
            { pattern: /\bsudo\s+su\b/i, reason: 'Privilege escalation (sudo su)' },
            { pattern: /\bdrop\s+database\b/i, reason: 'SQL database deletion' },
            { pattern: /\btruncate\s+table\b/i, reason: 'SQL table truncation' },
            { pattern: /\bdelete\s+from\s+\w+\s*;/i, reason: 'SQL delete without WHERE clause' },
        ];

        // Check built-in patterns
        for (const { pattern, reason } of builtInPatterns) {
            if (pattern.test(command)) {
                return reason;
            }
        }

        // Check user-configured dangerous patterns from settings
        const config = vscode.workspace.getConfiguration('victor.autonomy');
        const userPatterns: string[] = config.get('dangerousCommandPatterns', []);

        for (const pattern of userPatterns) {
            try {
                if (command.toLowerCase().includes(pattern.toLowerCase())) {
                    return `Matches custom pattern: "${pattern}"`;
                }
            } catch {
                // Ignore invalid patterns
            }
        }

        return null;
    }

    private _refresh(): void {
        this._onDidChangeTreeData.fire(undefined);
    }
}

/**
 * Register terminal agent commands
 */
export function registerTerminalAgentCommands(
    context: vscode.ExtensionContext,
    provider: TerminalAgentProvider
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.terminal.suggest', async () => {
            const intent = await vscode.window.showInputBox({
                prompt: 'What do you want to do?',
                placeHolder: 'e.g., Install dependencies, Run tests, Build project...',
            });

            if (intent) {
                const cmd = await provider.suggestCommand(intent);
                if (cmd && !cmd.isDangerous) {
                    // Auto-approve non-dangerous commands with user confirmation
                    const action = await vscode.window.showInformationMessage(
                        `Run: ${cmd.command}`,
                        'Run',
                        'Edit',
                        'Cancel'
                    );

                    if (action === 'Run') {
                        await provider.approveCommand(cmd.id);
                    } else if (action === 'Edit') {
                        const edited = await vscode.window.showInputBox({
                            value: cmd.command,
                            prompt: 'Edit command before running',
                        });
                        if (edited) {
                            cmd.command = edited;
                            await provider.approveCommand(cmd.id);
                        }
                    } else {
                        await provider.rejectCommand(cmd.id);
                    }
                }
            }
        }),

        vscode.commands.registerCommand('victor.terminal.run', async () => {
            const command = await vscode.window.showInputBox({
                prompt: 'Enter command to run',
                placeHolder: 'npm install, pytest, etc.',
            });

            if (command) {
                await provider.runCommand(command);
            }
        }),

        vscode.commands.registerCommand('victor.terminal.approve', async (item: TerminalTreeItem) => {
            if (item.itemType === 'command' && item.data) {
                const cmd = item.data as TerminalCommand;
                await provider.approveCommand(cmd.id);
            }
        }),

        vscode.commands.registerCommand('victor.terminal.reject', async (item: TerminalTreeItem) => {
            if (item.itemType === 'command' && item.data) {
                const cmd = item.data as TerminalCommand;
                await provider.rejectCommand(cmd.id);
            }
        }),

        vscode.commands.registerCommand('victor.terminal.history', () => {
            const history = provider.getCommandHistory();
            if (history.length === 0) {
                vscode.window.showInformationMessage('No command history for this terminal.');
                return;
            }

            vscode.window.showQuickPick(
                history.map(cmd => ({
                    label: cmd.command,
                    description: cmd.status,
                    detail: cmd.description,
                    command: cmd,
                })),
                { placeHolder: 'Select a command to run again' }
            ).then(async selected => {
                if (selected) {
                    await provider.runCommand(selected.command.command);
                }
            });
        })
    );
}
