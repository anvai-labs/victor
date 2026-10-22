/**
 * Git Panel View Provider
 *
 * Provides a tree view of Git repository status and operations:
 * - Current branch and tracking status
 * - Staged/unstaged/untracked files
 * - Commit history
 * - AI-assisted commit message generation
 */

import * as vscode from 'vscode';
import { getProviders } from './providerAccessor';

interface GitStatus {
    is_git_repo: boolean;
    branch?: string;
    tracking?: string;
    staged?: { status: string; file: string }[];
    unstaged?: { status: string; file: string }[];
    untracked?: string[];
    is_clean?: boolean;
    error?: string;
}

interface CommitInfo {
    hash: string;
    author: string;
    email: string;
    relative_date: string;
    message: string;
}

type GitItemType = 'branch' | 'category' | 'file' | 'commit' | 'action' | 'loading' | 'error' | 'info';

export class GitPanelViewProvider implements vscode.TreeDataProvider<GitItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<GitItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private isLoading = false;
    private status: GitStatus | null = null;
    private commits: CommitInfo[] = [];
    private lastError: string | null = null;

    constructor() {
        this.refresh();
    }

    async refresh(): Promise<void> {
        if (this.isLoading) {
            return;
        }

        this.isLoading = true;
        this.lastError = null;
        this._onDidChangeTreeData.fire(undefined);

        try {
            const providers = getProviders();
            if (!providers?.victorClient) {
                this.lastError = 'Victor client not available';
                return;
            }

            // Load status and log in parallel
            const [statusResult, logResult] = await Promise.all([
                providers.victorClient.getGitStatus(),
                providers.victorClient.getGitLog(10),
            ]);

            this.status = statusResult;
            this.commits = logResult.commits || [];

        } catch (error) {
            console.error('Failed to load git panel:', error);
            this.lastError = error instanceof Error ? error.message : 'Unknown error';
        } finally {
            this.isLoading = false;
            this._onDidChangeTreeData.fire(undefined);
        }
    }

    getTreeItem(element: GitItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: GitItem): GitItem[] {
        if (!element) {
            // Root level
            if (this.isLoading) {
                return [new GitItem('Loading git status...', 'loading', vscode.TreeItemCollapsibleState.None)];
            }

            if (this.lastError) {
                return [new GitItem(`Error: ${this.lastError}`, 'error', vscode.TreeItemCollapsibleState.None)];
            }

            if (!this.status?.is_git_repo) {
                return [new GitItem('Not a git repository', 'info', vscode.TreeItemCollapsibleState.None)];
            }

            const items: GitItem[] = [];

            // Branch info
            const branchLabel = this.status.tracking
                ? `${this.status.branch} → ${this.status.tracking}`
                : this.status.branch || 'unknown';
            items.push(new GitItem(
                branchLabel,
                'branch',
                vscode.TreeItemCollapsibleState.None,
                undefined,
                'git-branch'
            ));

            // Staged changes
            if (this.status.staged && this.status.staged.length > 0) {
                items.push(new GitItem(
                    `Staged Changes (${this.status.staged.length})`,
                    'category',
                    vscode.TreeItemCollapsibleState.Expanded,
                    { type: 'staged', files: this.status.staged },
                    'diff-added'
                ));
            }

            // Unstaged changes
            if (this.status.unstaged && this.status.unstaged.length > 0) {
                items.push(new GitItem(
                    `Unstaged Changes (${this.status.unstaged.length})`,
                    'category',
                    vscode.TreeItemCollapsibleState.Expanded,
                    { type: 'unstaged', files: this.status.unstaged },
                    'diff-modified'
                ));
            }

            // Untracked files
            if (this.status.untracked && this.status.untracked.length > 0) {
                items.push(new GitItem(
                    `Untracked Files (${this.status.untracked.length})`,
                    'category',
                    vscode.TreeItemCollapsibleState.Collapsed,
                    { type: 'untracked', files: this.status.untracked },
                    'question'
                ));
            }

            // Clean status message
            if (this.status.is_clean) {
                items.push(new GitItem(
                    'Working tree clean',
                    'info',
                    vscode.TreeItemCollapsibleState.None,
                    undefined,
                    'check'
                ));
            }

            // Commit history
            if (this.commits.length > 0) {
                items.push(new GitItem(
                    'Recent Commits',
                    'category',
                    vscode.TreeItemCollapsibleState.Collapsed,
                    { type: 'commits' },
                    'history'
                ));
            }

            // Actions
            items.push(new GitItem(
                'AI Commit',
                'action',
                vscode.TreeItemCollapsibleState.None,
                { action: 'ai_commit' },
                'sparkle'
            ));

            return items;
        }

        // Children based on category
        if (element.data?.type === 'staged') {
            const files = element.data.files as { status: string; file: string }[];
            return this.getFileChildren(files, 'staged');
        }

        if (element.data?.type === 'unstaged') {
            const files = element.data.files as { status: string; file: string }[];
            return this.getFileChildren(files, 'unstaged');
        }

        if (element.data?.type === 'untracked') {
            const files = element.data.files as string[];
            return files.map((file: string) =>
                new GitItem(file, 'file', vscode.TreeItemCollapsibleState.None, { file, status: '?' }, 'file')
            );
        }

        if (element.data?.type === 'commits') {
            return this.commits.slice(0, 10).map(commit => {
                const item = new GitItem(
                    commit.message.substring(0, 60),
                    'commit',
                    vscode.TreeItemCollapsibleState.None,
                    commit as unknown as Record<string, unknown>,
                    'git-commit'
                );
                item.description = `${commit.author} • ${commit.relative_date}`;
                item.tooltip = `${commit.hash.substring(0, 7)} - ${commit.message}\n\nBy ${commit.author} (${commit.email})\n${commit.relative_date}`;
                return item;
            });
        }

        return [];
    }

    private getFileChildren(
        files: { status: string; file: string }[],
        changeType: 'staged' | 'unstaged'
    ): GitItem[] {
        return files.map(f => {
            const statusIcon = this.getStatusIcon(f.status);
            const item = new GitItem(
                f.file,
                'file',
                vscode.TreeItemCollapsibleState.None,
                { file: f.file, status: f.status, changeType },
                statusIcon
            );
            item.description = this.getStatusLabel(f.status);
            item.command = {
                command: 'vscode.diff',
                title: 'Show Diff',
                arguments: changeType === 'staged'
                    ? [vscode.Uri.parse(`git:${f.file}~HEAD`), vscode.Uri.file(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath + '/' + f.file), `${f.file} (staged)`]
                    : [vscode.Uri.file(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath + '/' + f.file), vscode.Uri.file(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath + '/' + f.file), f.file]
            };
            return item;
        });
    }

    private getStatusIcon(status: string): string {
        const icons: Record<string, string> = {
            'M': 'diff-modified',
            'A': 'diff-added',
            'D': 'diff-removed',
            'R': 'diff-renamed',
            'C': 'diff-added',
            '?': 'question',
        };
        return icons[status] || 'file';
    }

    private getStatusLabel(status: string): string {
        const labels: Record<string, string> = {
            'M': 'modified',
            'A': 'added',
            'D': 'deleted',
            'R': 'renamed',
            'C': 'copied',
            '?': 'untracked',
        };
        return labels[status] || status;
    }
}

class GitItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly itemType: GitItemType,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly data?: Record<string, unknown>,
        iconName?: string
    ) {
        super(label, collapsibleState);

        if (iconName) {
            const color = itemType === 'error' ? new vscode.ThemeColor('errorForeground')
                : itemType === 'action' ? new vscode.ThemeColor('textLink.foreground')
                : undefined;
            this.iconPath = new vscode.ThemeIcon(iconName, color);
        }

        switch (itemType) {
            case 'loading':
                this.iconPath = new vscode.ThemeIcon('loading~spin');
                break;
            case 'error':
                this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
                break;
            case 'action':
                this.contextValue = 'action';
                if (data?.action === 'ai_commit') {
                    this.command = {
                        command: 'victor.aiCommit',
                        title: 'Create AI Commit',
                    };
                }
                break;
            case 'branch':
                this.contextValue = 'branch';
                break;
            case 'file':
                this.contextValue = data?.changeType === 'staged' ? 'stagedFile' : 'file';
                break;
        }
    }
}

/**
 * Register Git panel commands
 */
export function registerGitPanelCommands(
    context: vscode.ExtensionContext,
    provider: GitPanelViewProvider
): void {
    // Refresh git panel
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshGitPanel', () => {
            provider.refresh();
        })
    );

    // AI-assisted commit
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.aiCommit', async () => {
            const providers = getProviders();
            if (!providers?.victorClient) {
                vscode.window.showWarningMessage('Victor client not available');
                return;
            }

            // Check if there are staged changes
            const status = await providers.victorClient.getGitStatus();
            if (!status.is_git_repo) {
                vscode.window.showWarningMessage('Not a git repository');
                return;
            }

            if (!status.staged || status.staged.length === 0) {
                const stageAll = await vscode.window.showQuickPick(['Yes', 'No'], {
                    placeHolder: 'No staged changes. Stage all changes?',
                });
                if (stageAll !== 'Yes') {
                    return;
                }
                // User wants to stage all - will be handled by backend
            }

            // Ask user how to generate commit message
            const commitOption = await vscode.window.showQuickPick([
                { label: 'Generate AI Commit Message', value: 'ai' },
                { label: 'Enter Manual Message', value: 'manual' },
            ], {
                placeHolder: 'How would you like to create the commit message?',
            });

            if (!commitOption) {
                return;
            }

            let message: string | undefined;
            if (commitOption.value === 'manual') {
                message = await vscode.window.showInputBox({
                    prompt: 'Enter commit message',
                    placeHolder: 'feat: Add new feature',
                });
                if (!message) {
                    return;
                }
            }

            // Perform commit with progress
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: commitOption.value === 'ai' ? 'Generating AI commit message...' : 'Creating commit...',
                cancellable: false,
            }, async () => {
                try {
                    const result = await providers.victorClient.gitCommit({
                        message,
                        use_ai: commitOption.value === 'ai',
                    });

                    if (result.success) {
                        vscode.window.showInformationMessage(`Committed: ${result.message?.substring(0, 50)}...`);
                        provider.refresh();
                    } else {
                        vscode.window.showErrorMessage(`Commit failed: ${result.error}`);
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Commit error: ${error}`);
                }
            });
        })
    );

    // Stage file
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.stageFile', async (item: GitItem) => {
            if (item.data?.file) {
                const terminal = vscode.window.createTerminal('Git');
                terminal.sendText(`git add "${item.data.file}"`);
                terminal.show();
                setTimeout(() => provider.refresh(), 1000);
            }
        })
    );

    // Unstage file
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.unstageFile', async (item: GitItem) => {
            if (item.data?.file) {
                const terminal = vscode.window.createTerminal('Git');
                terminal.sendText(`git reset HEAD "${item.data.file}"`);
                terminal.show();
                setTimeout(() => provider.refresh(), 1000);
            }
        })
    );

    // Show diff
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showGitDiff', async () => {
            const providers = getProviders();
            if (!providers?.victorClient) {
                vscode.window.showWarningMessage('Victor client not available');
                return;
            }

            const diffOption = await vscode.window.showQuickPick([
                { label: 'Staged Changes', value: 'staged' },
                { label: 'Unstaged Changes', value: 'unstaged' },
            ], {
                placeHolder: 'Which diff to show?',
            });

            if (!diffOption) {
                return;
            }

            const result = await providers.victorClient.getGitDiff({
                staged: diffOption.value === 'staged',
            });

            if (result.diff) {
                const doc = await vscode.workspace.openTextDocument({
                    content: result.diff,
                    language: 'diff',
                });
                await vscode.window.showTextDocument(doc);
            } else {
                vscode.window.showInformationMessage('No changes to show');
            }
        })
    );
}
